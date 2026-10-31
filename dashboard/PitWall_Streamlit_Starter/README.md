# PitWall Streamlit Dashboard

This is the first working user-interface milestone for the EE-322L PitWall proposal.
It matches the proposed dashboard concept, with a synthetic demo mode for development
before the radio receiver is integrated.

## Run on your laptop (VS Code terminal)

```bash
python -m venv .venv
```

Activate the virtual environment:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the localhost URL displayed in the terminal (usually http://localhost:8501).

## Pages

1. Live Dashboard: left/right wheel RPM, motor command, RF packet recovery,
   acceleration if present, and latest valid packet.
2. RF Link Monitor: packet-recovery rate and packet validity.
3. ML Prediction: a reserved chart for the proposal's 0.5-second wheel-speed forecast.
4. Data Log: inspection and CSV download.
5. Driver Comparison: preliminary wheel-speed comparison if driver IDs exist.
6. Settings: data schema and integration notes.

## Integration contract

Use **Upload decoded CSV** in the sidebar to load receiver output. Required CSV columns:

`timestamp_ms,seq,left_rpm,right_rpm,throttle_pct,crc_ok`

Optional: `ax,ay,az,yaw_dps,predicted_rpm,driver,lap`.

Example:

```csv
timestamp_ms,seq,left_rpm,right_rpm,throttle_pct,crc_ok
100,1,96,98,50,true
200,2,100,102,54,true
300,3,,,,false
400,4,110,114,60,true
```

The radio receiver must validate CRC *before* telemetry is considered valid.
Packet sequence gaps allow PRR estimation, but use a known experiment start/end
range for formal evaluation so losses at both boundaries are counted.

Notes:
- The demo generates artificial telemetry; it is **not** hardware reception.
- Synthetic `predicted_rpm` is illustrative only; it is **not** a trained model.
- The 19-byte proposal telemetry packet has limited room. MPU6050 data may only
  exist in the ESP32 local log until the RF protocol is explicitly extended.
- The proposal's ML task is **0.5-second wheel-speed prediction**, not anomaly detection.
- The current CSV view is a recorded-file view. For real live input, keep the SDR
  decoding process independent and add a queue/file tail or local IPC reader.
