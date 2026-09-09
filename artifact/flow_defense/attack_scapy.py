"""Physical realisability check for the semantic triggers, using scapy.

This module is not used for training-time poisoning. It answers a separate
question: is a trigger defined as a perturbation of flow statistics actually
producible by real packets? For each trigger feature it constructs a packet
sequence whose measured statistics land in the intended range, so a trigger
cannot be dismissed as an artifact of the feature extractor.

Nothing in claims/ imports this module, and scapy is an optional dependency.

Usage:
    python -m flow_defense.attack_scapy --output triggers.pcap --kind high_ct_dst_sport
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import List, Optional


def _try_import_scapy():
    try:
        from scapy.all import IP, TCP, UDP, Raw, Ether, wrpcap  # noqa: F401
        from scapy.all import PacketList  # noqa: F401
        return True
    except ImportError:
        return False


def generate_high_ct_dst_sport_packets(
    target_ip: str = "10.0.0.1",
    target_port: int = 443,
    num_connections: int = 500,
    duration_sec: float = 2.0,
):
    """Many TCP SYN packets from one source to one destination port.

    Typical of port scanning or DDoS reconnaissance, and it drives
    ct_dst_sport_ltm very high. Source ports are randomised, as the protocol
    """
    from scapy.all import IP, TCP
    packets = []
    gap = duration_sec / max(num_connections, 1)
    for i in range(num_connections):
        sport = random.randint(1024, 65535)
        pkt = IP(src="192.168.1.100", dst=target_ip) / TCP(
            sport=sport,
            dport=target_port,
            flags="S",
            seq=random.randint(0, 2**31),
        )
        pkt.time = time.time() + i * gap
        packets.append(pkt)
    return packets


def generate_slowloris_like_packets(
    target_ip: str = "10.0.0.2",
    target_port: int = 80,
    num_connections: int = 100,
    slow_interval_sec: float = 15.0,
):
    """Slowloris-style slow HTTP: a legitimate header with abnormal timing.

    A well-formed GET, but subsequent packets arrive every 15 seconds, holding
    the connection open without generating much traffic. This is the clearest
    example of the legitimate-header, abnormal-timing pattern.
    """
    from scapy.all import IP, TCP, Raw
    packets = []
    base_time = time.time()
    for i in range(num_connections):
        sport = 1024 + i
        conn_start = base_time + i * 0.1
        syn = IP(src="192.168.1.100", dst=target_ip) / TCP(
            sport=sport, dport=target_port, flags="S", seq=1000 + i,
        )
        syn.time = conn_start
        packets.append(syn)
        partial_request = (
            b"GET / HTTP/1.1\r\n"
            b"Host: example.com\r\n"
            b"User-Agent: Mozilla/5.0\r\n"
        )
        for chunk_idx in range(6):
            extra_header = f"X-Custom-{chunk_idx}: keep-alive\r\n".encode()
            pkt = IP(src="192.168.1.100", dst=target_ip) / TCP(
                sport=sport, dport=target_port, flags="PA",
                seq=1001 + i + chunk_idx, ack=1,
            ) / Raw(load=extra_header)
            pkt.time = conn_start + (chunk_idx + 1) * slow_interval_sec
            packets.append(pkt)
    return packets


def generate_low_dload_packets(
    target_ip: str = "10.0.0.3",
    target_port: int = 21,
    num_packets: int = 200,
):
    """A session with almost no download-direction payload, matching dload at -5 sigma.

    The client connects and uploads while the server barely responds, as in an
    anomalous FTP command channel, so measured dload falls far below normal.
    """
    from scapy.all import IP, TCP, Raw
    packets = []
    base = time.time()
    for i in range(num_packets):
        sport = 1024 + (i % 100)
        pkt = IP(src="192.168.1.100", dst=target_ip) / TCP(
            sport=sport, dport=target_port, flags="PA", seq=i * 10, ack=1,
        ) / Raw(load=b"PWD\r\n")
        pkt.time = base + i * 0.05
        packets.append(pkt)
    return packets


TRIGGER_GENERATORS = {
    "high_ct_dst_sport": generate_high_ct_dst_sport_packets,
    "slowloris": generate_slowloris_like_packets,
    "low_dload": generate_low_dload_packets,
}


def build_composite_trigger_pcap(output_path: str) -> None:
    """Emit one PCAP containing every trigger, for independent verification.

    Writes one PCAP holding several representative sub-sessions:
      - high connection count, matching the ct_dst_sport_ltm trigger
      - Slowloris slow HTTP, matching the rate and timing triggers together
      - low-download FTP command session, matching the dload trigger
    """
    if not _try_import_scapy():
        raise ImportError(
            "scapy is not installed. Install with: pip install scapy\n"
            "Then re-run: python -m flow_defense.attack_scapy --output triggers.pcap"
        )
    from scapy.all import wrpcap
    all_packets = []
    for kind, gen in TRIGGER_GENERATORS.items():
        print(f"[scapy] Generating {kind} ...")
        all_packets.extend(gen())
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    wrpcap(str(out), all_packets)
    print(f"[scapy] Wrote {len(all_packets)} packets to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scapy-based trigger PCAP generator")
    parser.add_argument(
        "--output", type=str, default="outputs/trigger_samples.pcap",
        help="Output PCAP file path",
    )
    parser.add_argument(
        "--kind", type=str, default="composite",
        choices=list(TRIGGER_GENERATORS.keys()) + ["composite"],
        help="Which trigger kind to generate (composite = all of them)",
    )
    args = parser.parse_args()

    if args.kind == "composite":
        build_composite_trigger_pcap(args.output)
    else:
        if not _try_import_scapy():
            raise ImportError("scapy is not installed. pip install scapy")
        from scapy.all import wrpcap
        packets = TRIGGER_GENERATORS[args.kind]()
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        wrpcap(args.output, packets)
        print(f"Wrote {len(packets)} packets to {args.output}")


if __name__ == "__main__":
    main()
