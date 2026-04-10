# IMU Sensors for Figure Skating — Research
**Date:** 2026-04-09
**Sources:** Claude Code research + Grok research + Ozon product pages

---

## Decision: Ordered 2x WitMotion WT9011DCL

- **Price:** ~1190 ₽ each (~$13), pair = ~2400 ₽ (~$26)
- **IMU:** 9-axis (MPU9250: accel + gyro + mag) + Kalman filter
- **Frequency:** Up to 200Hz (configurable, default 10Hz)
- **BLE:** 5.0
- **Battery:** 130 mAh → 6-8 hours
- **Size:** 23.5 × 32.5 × 11.4 mm, 15g
- **Charge:** Type-C
- **Protocol:** WIT Standard Protocol (open, documented)

### Also available: Seeeduino XIAO BLE Sense (nRF52840)
- **IMU:** 6-axis only (LSM6DS3: accel + gyro, no magnetometer)
- **BLE:** 5.0 with full GATT control
- **Size:** 21 × 17.5 mm (even smaller)
- **Extras:** Built-in PDM microphone, TinyML support, BQ25101 battery charger on board
- **Needs:** Li-Po battery + firmware (Arduino/CircuitPython) + 3D-printed case
- **Use case:** 2nd-gen custom sensor after WT901 validation

---

## Architecture Decision

### Problem: iOS Safari does NOT support Web Bluetooth
Coaches at ice rinks use iPhones. Web Bluetooth is Chrome/Edge only.

### Recommended: Hybrid approach

```
Phase 1 (MVP): WT9011DCL → Chrome Android → Web Bluetooth → WebSocket → FastAPI
Phase 2 (iOS): Minimal React Native companion app (react-native-ble-plx) → WebSocket → FastAPI
```

### Data Flow

```
IMU on skates --BLE--> [Phone/Browser] --WebSocket--> [FastAPI Backend]
                                              |
                                        Video (getUserMedia)
                                              |
                                        Post-hoc sync via timestamps
```

### Sync Strategy
- `performance.now()` timestamps on both IMU packets and video frames
- Single "Start Recording" button starts both simultaneously
- Post-hoc: cross-correlation of jump detection peaks (IMU accel) + video (CoM)
- Optional: clap/tap sync signal at start for precise alignment

---

## Mounting on Skates

### Best position: Heel counter (задник ботинка)
- Rigid coupling to blade = accurate rotation measurement
- Tanaka 2023 confirmed: foot yaw is primary blade edge feature
- Secondary: lateral ankle (above boot collar) — easier to mount

### How to mount (removable, not permanent)
- Velcro strap (2-3cm wide) + EVA foam pad under sensor (vibration isolation)
- 3D-printed clip/harness (adapt from Thingiverse/Printables IMU mounts)
- Silicone case or heat-shrink for moisture protection

### Don't mount on
- Blade itself (extreme vibration from ice)
- Toe (interferes with landings)
- Too high on shin (less foot data)

---

## What IMU Data Adds to Our Pipeline

| Feature | Method | Source |
|---------|--------|--------|
| Rotation counting | Euler angle integration via scipy Rotation | Synergie/AIOnIce |
| Blade edge detection | Foot yaw rotation (primary feature) | JudgeAI-LutzEdge (Tanaka 2023) |
| Phase detection | Jerk = d²(gyro)/dt² threshold | Synergie (2nd derivative method) |
| Jump classification | Transformer on (400, 6) IMU windows | Synergie (7 classes, 93%+) |
| Video sync | Cross-correlation of IMU accel peaks + video CoM | Custom |

---

## Protocol: WT901 BLE (WIT Standard Protocol)

### GATT Service
- Nordic UART Service (de facto standard): `6E400001-B5A3-F393-E0A9-E50E24DCCA9E`
- TX characteristic: `6E400002-B5A3-F393-E0A9-E50E24DCCA9E`
- RX characteristic: `6E400003-B5A3-F393-E0A9-E50E24DCCA9E`

### Data Format
- Packets start with `0x55`, then type byte, then payload
- Types: accel, gyro, angle (Euler), quaternion, magnetometer
- At 100Hz with quaternions: ~30 bytes/packet, trivial for BLE

### Bandwidth
- 9-axis @ 100Hz, 2 sensors = ~9.6 KB/s = ~76.8 kbps
- BLE 4.2 max: ~1-2 Mbps — more than sufficient
- Packet loss at ice rink: <1% expected (open space, low interference)

### Linux Compatibility
- Works via Chrome + BlueZ (standard Linux BT stack)
- No proprietary drivers needed
- If issues: `sudo systemctl restart bluetooth`

---

## Alternatives Evaluated

| Variant | Price/pair | Soldering | IMU | For us? |
|---------|-----------|-----------|-----|---------|
| **WT9011DCL** (ordered) | ~$26 | No | 9-axis + Kalman | **Best for quick start** |
| XIAO BLE Sense (at home) | ~$37 | Min (battery) | 6-axis | 2nd-gen custom |
| LILYGO T-OI Plus + T-ICM | ~$21-30 | Yes | 9-axis (ICM-20948) | DIY, proven by Mesquite mocap |
| nRF52840 + BNO085 | ~$45 | Yes | 9-axis (best fusion) | Top accuracy, complex |
| Arduino Nano 33 BLE Sense | ~$72 | No | 9-axis (LSM9DS1) | Expensive, older sensor |
| Xsens DOT (reference) | ~$1000 | No | Reference | Research only |
| Phone as IMU ($0) | $0 | No | 16-bit, 100-400Hz | **Validate pipeline first** |

---

## Open-Source References

| Project | URL | Relevance |
|---------|-----|-----------|
| Mesquite mocap | github.com/Mesquite-Mocap/mesquite.cc | WebXR IMU mocap, T-OI Plus BOM, firmware |
| Quatro | github.com/tejaswigowda/quatro | Web Bluetooth IMU visualizer (ESP32 + ICM-20948) |
| Synergie/AIOnIce | github.com/Mart1t1/Synergie | Skating jump classification from IMU (Xsens + FastAPI + Keras) |
| JudgeAI-LutzEdge | github.com/ryota-skating/JudgeAI-LutzEdge | Blade edge detection using IMU rotation |
| bleak | github.com/hbldh/bleak | Best Python BLE library (asyncio, cross-platform) |
| MotioSuit | github.com/alvaroferran/MotioSuit | Open-source IMU mocap suit (BNO055 + BLE) |
| pywitmotion | GitHub | Python parser for WitMotion BLE protocol |

---

## Next Steps

1. [ ] Receive WT9011DCL, charge, test via official WitMotion Android app
2. [ ] Connect via Chrome on Linux/Android — verify Web Bluetooth works
3. [ ] Write minimal Web Bluetooth JS (connect + parse quaternion packets)
4. [ ] Sync with video recording (getUserMedia + performance.now timestamps)
5. [ ] Add WebSocket relay to FastAPI backend
6. [ ] Validate: compare IMU jump detection with video CoM-based detection
7. [ ] Later: React Native companion app for iOS support
8. [ ] Later: experiment with XIAO BLE Sense as custom 2nd-gen sensor
