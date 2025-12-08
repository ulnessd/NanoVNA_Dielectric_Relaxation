#!/usr/bin/env python3
import serial
import time
import sys
from serial.tools import list_ports


def find_nanovna_port():
    """Scans for potential NanoVNA ports."""
    print("Scanning ports...")
    ports = list(list_ports.comports())
    candidates = []
    for p in ports:
        print(f"  Found: {p.device} - {p.description}")
        # Typical identifiers for NanoVNA
        if "STMicroelectronics" in p.description or "ChibiOS" in p.description or "ACM" in p.device:
            candidates.append(p.device)
    return candidates


def test_connection(port):
    """Attempts to connect and get version info."""
    print(f"\nAttempting connection to {port}...")
    try:
        # standard NanoVNA baud rate is often ignored by USB-CDC, but 115200 is safe
        ser = serial.Serial(port, baudrate=115200, timeout=2)

        # Clear any startup noise
        ser.read_all()

        # Send 'info' or 'version' command
        # \r is the termination character it expects
        ser.write(b"version\r")
        time.sleep(0.5)  # Give it a moment to reply

        response = ser.read_all().decode(errors='ignore')

        if response:
            print("--- SUCCESS! NanoVNA Responded ---")
            print(f"Raw Response:\n{response.strip()}")
            print("----------------------------------")
            return True
        else:
            print("Connected, but no response received.")
            return False

    except serial.SerialException as e:
        print(f"Failed to open port: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()


if __name__ == "__main__":
    potential_ports = find_nanovna_port()

    if not potential_ports:
        print("\nNo obvious NanoVNA ports found.")
        print("Tip: Check USB cable and ensure device is ON.")
        # Fallback for some Linux configs where description is blank
        potential_ports = ['/dev/ttyACM0', '/dev/ttyACM1']

    for p in potential_ports:
        if test_connection(p):
            break