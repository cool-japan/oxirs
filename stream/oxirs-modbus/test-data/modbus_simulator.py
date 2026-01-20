#!/usr/bin/env python3
"""
Modbus TCP Simulator for OxiRS Testing

This script provides a Modbus TCP server that simulates industrial devices
for testing the oxirs-modbus client.

Usage:
    # Install pymodbus
    pip install pymodbus

    # Run simulator
    python modbus_simulator.py

    # Run with custom port
    python modbus_simulator.py --port 5020

Features:
    - Holding registers (FC 0x03): 100 registers starting at address 0
    - Input registers (FC 0x04): 50 registers with simulated sensor data
    - Coils (FC 0x01): 32 coils
    - Write single register (FC 0x06)
    - Write multiple registers (FC 0x10)

The simulator pre-populates registers with test data that the oxirs-modbus
integration tests can verify.
"""

import argparse
import logging
import signal
import sys
from threading import Thread
import time

try:
    from pymodbus.server import StartTcpServer
    from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext
    from pymodbus.datastore import ModbusSequentialDataBlock
    from pymodbus.version import version as pymodbus_version
except ImportError:
    print("Error: pymodbus not installed. Install with: pip install pymodbus")
    sys.exit(1)

# Configure logging
logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.INFO)

# Test data patterns
HOLDING_REGISTERS = [
    100, 200, 300, 400, 500,    # Registers 0-4: Simple incrementing
    1000, 2000, 3000, 4000,     # Registers 5-8: Larger values
    0xFFFF, 0x8000, 0x0001,     # Registers 9-11: Edge cases
    0x1234, 0x5678, 0x9ABC,     # Registers 12-14: Hex patterns
] + [i * 10 for i in range(85)]  # Fill rest with pattern

INPUT_REGISTERS = [
    225,   # Register 0: Temperature (22.5°C * 10)
    501,   # Register 1: Humidity (50.1%)
    1013,  # Register 2: Pressure (1013 hPa)
    3300,  # Register 3: Voltage (330.0V * 10)
    150,   # Register 4: Current (15.0A * 10)
] + [0] * 45  # Rest are zeros (unused sensors)

COILS = [True, False, True, True, False, True, False, False] + [False] * 24


class ModbusSimulator:
    """Modbus TCP Simulator for testing"""

    def __init__(self, host: str = "127.0.0.1", port: int = 502):
        self.host = host
        self.port = port
        self.running = False
        self.server_thread = None

    def create_context(self) -> ModbusServerContext:
        """Create Modbus data context with test data"""

        # Create data blocks
        # Note: ModbusSequentialDataBlock uses 1-based addressing internally
        coils = ModbusSequentialDataBlock(0, [False] + COILS)
        discrete_inputs = ModbusSequentialDataBlock(0, [False] * 33)
        input_registers = ModbusSequentialDataBlock(0, [0] + INPUT_REGISTERS)
        holding_registers = ModbusSequentialDataBlock(0, [0] + HOLDING_REGISTERS)

        # Create slave context
        store = ModbusSlaveContext(
            di=discrete_inputs,
            co=coils,
            hr=holding_registers,
            ir=input_registers
        )

        # Create server context (single slave, unit_id=1)
        context = ModbusServerContext(slaves=store, single=True)

        return context

    def start(self):
        """Start the Modbus TCP server"""
        log.info(f"Starting Modbus TCP Simulator on {self.host}:{self.port}")
        log.info(f"pymodbus version: {pymodbus_version}")
        log.info("")
        log.info("Test data available:")
        log.info(f"  - Holding registers (FC 0x03): {len(HOLDING_REGISTERS)} registers")
        log.info(f"  - Input registers (FC 0x04): {len(INPUT_REGISTERS)} registers")
        log.info(f"  - Coils (FC 0x01): {len(COILS)} coils")
        log.info("")
        log.info("Sample holding register values:")
        log.info(f"  Registers 0-4: {HOLDING_REGISTERS[:5]}")
        log.info("")
        log.info("Sample input register values (sensors):")
        log.info(f"  Register 0: {INPUT_REGISTERS[0] / 10}°C (temperature)")
        log.info(f"  Register 1: {INPUT_REGISTERS[1] / 10}% (humidity)")
        log.info(f"  Register 2: {INPUT_REGISTERS[2]} hPa (pressure)")
        log.info("")
        log.info("Press Ctrl+C to stop")

        context = self.create_context()

        try:
            StartTcpServer(
                context=context,
                address=(self.host, self.port)
            )
        except KeyboardInterrupt:
            log.info("Shutting down simulator...")
        except Exception as e:
            log.error(f"Server error: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Modbus TCP Simulator for OxiRS testing"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address to bind (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5020,  # Use non-privileged port for testing
        help="Port to listen on (default: 5020)"
    )

    args = parser.parse_args()

    simulator = ModbusSimulator(host=args.host, port=args.port)
    simulator.start()


if __name__ == "__main__":
    main()
