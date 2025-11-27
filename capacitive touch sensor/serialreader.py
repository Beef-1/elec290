import serial
import time

try:
    # Open the serial port (adjust 'COM5' and baud rate as needed)
    ser = serial.Serial('COM5', 9600, timeout=1) 
    print(f"Serial port {ser.name} opened successfully.")
    print("Reading data... Press Ctrl+C to stop.")
    print("-" * 50)

    # Continuously read and print all received data
    while True:
        if ser.in_waiting > 0:
            # Read all available data
            received_data = ser.read(ser.in_waiting)
            try:
                # Try to decode as text
                decoded_data = received_data.decode('utf-8', errors='replace')
                print(decoded_data, end='', flush=True)
            except:
                # If decoding fails, print as hex
                print(f"[RAW: {received_data.hex()}]", end='', flush=True)
        time.sleep(0.01)  # Small delay to prevent excessive CPU usage

except KeyboardInterrupt:
    print("\n\nStopped by user.")

except serial.SerialException as e:
    print(f"Error: {e}")

finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("Serial port closed.")