# BLE IMU Web Integration Research
**Date:** 2026-04-09
**Goal:** Technical feasibility of connecting Bluetooth IMU sensors to a web application (React + FastAPI) for real-time figure skating analysis

---

## Executive Summary

BLE IMU integration into a web application is **technically feasible but architecturally constrained**. The critical blocker is **iOS Safari does not support Web Bluetooth** and likely never will. This makes Option A (pure browser-based) a non-starter for ice rink use where coaches use iPhones. The recommended architecture is **Option B (native companion app)** using React Native, which gives full BLE access on both iOS and Android while reusing existing React/TypeScript expertise.

The phone-as-IMU approach (Option D) is the **lowest-friction starting point** -- zero hardware cost, immediate deployment, and modern phone IMUs are excellent. Dedicated BLE IMUs (Movesense, custom ESP32) provide better mounting options and multi-sensor fusion but add hardware complexity.

---

## 1. Web Bluetooth API

### 1.1 Browser Support Matrix (as of 2026-04)

| Browser | Desktop | Mobile | Notes |
|---------|---------|--------|-------|
| **Chrome** | Yes (53+) | Yes (Android 6+) | **Primary platform.** Full GATT support, notifications, reliable streaming |
| **Edge** | Yes (79+) | Yes (Android) | Chromium-based, same as Chrome |
| **Firefox** | **NO** | **NO** | Not implemented. No plans (privacy concerns) |
| **Safari (macOS)** | **NO** | N/A | Apple has not implemented Web Bluetooth on desktop either |
| **Safari (iOS)** | N/A | **NO** | **CRITICAL BLOCKER.** Apple explicitly blocks Web Bluetooth on iOS. No indication this will change |
| **Samsung Internet** | N/A | Yes | Chromium-based on Android |
| **Chrome on iOS** | N/A | **NO** | Uses WKWebView on iOS, which lacks Web Bluetooth |

**Bottom line:** Web Bluetooth works on **Chrome/Edge on desktop and Android only**. iOS is completely unsupported.

### 1.2 Streaming Performance

- **Notification-based streaming** (GATT characteristic notifications) is the standard pattern for high-frequency data
- Typical achievable rate: **50-100Hz** for small payloads (10-20 bytes per notification) in practice
- **200Hz is at the practical limit** of Web Bluetooth notification delivery in JavaScript
- Latency: **20-100ms** from device to JS callback (varies by OS, Bluetooth stack, CPU load)
- **Connection interval:** BLE 4.2 default is 7.5-30ms. Can negotiate down to 7.5ms for higher throughput
- **No guaranteed delivery:** BLE notifications are "fire and forget". Packet loss requires application-level handling

### 1.3 Security Model

- **HTTPS required** (or localhost for development)
- **User gesture required** for `navigator.bluetooth.requestDevice()` -- must be triggered by button click or similar
- **No background scanning** -- page must be in foreground
- **Per-origin device access** -- each website gets its own device allowlist
- **Auto-reconnect** available after initial pairing via `device.gatt.connect()`

### 1.4 Known Limitations

- **iOS Safari: completely unsupported** (deal-breaker for ice rink use)
- **Connection timeout:** Most browsers disconnect after ~30 minutes of inactivity. Notifications keep connection alive
- **Single device limit on some platforms:** iOS Web Bluetooth not available, but on Android Chrome, can connect to multiple devices
- **MTU size:** Default 23 bytes (20 payload). Can negotiate up to 512 bytes on BLE 4.2+, but browser support varies (Chrome Android: up to 517 bytes)
- **No service discovery caching** across page reloads in some implementations

### 1.5 GATT Service/Characteristic Conventions for IMU

There is **no standardized Bluetooth SIG service** specifically for 9-axis IMU data. Most BLE IMU sensors use vendor-specific services. Common patterns:

**Common Vendor GATT Structures:**

```
# Movesense (Suunto) - most popular skating IMU
Service: 00001826-... (custom)
  Characteristic: Accelerometer (0x2A6E or custom UUID)
  Characteristic: Gyroscope (0x2A50 or custom UUID)
  Characteristic: Magnetometer (0x2A71 or custom UUID)
  Characteristic: Quaternion (fused orientation)

# Generic IMU pattern (most BLE IMUs follow this)
Service: Vendor-specific UUID
  Characteristic: XYZ accel (6 bytes: 3x int16, ±16g, 1mg resolution)
  Characteristic: XYZ gyro (6 bytes: 3x int16, ±2000dps, 61mdps resolution)
  Characteristic: XYZ mag (6 bytes: 3x int16)
  OR: Combined 9-axis (18 bytes per sample)

# Alternative: Nordic UART Service (NUS) for raw byte streaming
Service: 6E400001-B5A3-F393-E0A9-E50E24DCCA9E (Nordic)
  Characteristic: TX (write) - for commands
  Characteristic: RX (notify) - for data stream
```

**Bluetooth SIG standardized services (partial match):**
- `0x181A` (Environmental Sensing) -- temperature, humidity (not IMU)
- `0x2A6E` (Acceleration Measurement) -- individual characteristic, not a full service
- `0x2A50` (Magnetic Flux Density) -- magnetometer only

**Practical implication:** Every BLE IMU has a **proprietary GATT profile**. You need per-device firmware knowledge or documentation to parse data.

---

## 2. Data Relay Architecture Comparison

### Option A: Browser -> WebSocket -> FastAPI (Web Bluetooth)

```
[Phone/Tablet Browser] --Web Bluetooth--> [JS: BLE notifications]
  --> [JS: WebSocket client] --> [FastAPI WebSocket endpoint]
  --> [processing pipeline]
```

**Pros:**
- Zero installation (web app only)
- Works with existing React frontend
- Direct browser-to-server communication

**Cons:**
- **iOS Safari: NO Web Bluetooth** (deal-breaker)
- 50-100Hz practical limit in browser
- Phone must stay on browser page (no background)
- User gesture required each session for device pairing
- JavaScript single-threaded GC pauses can cause jitter

**Verdict: NOT VIABLE** for ice rink deployment (iOS coaches)

---

### Option B: Native Companion App -> WebSocket -> FastAPI (RECOMMENDED)

```
[React Native App] --Native BLE--> [JS: BLE notifications @100Hz]
  --> [JS: WebSocket client] --> [FastAPI WebSocket endpoint]
  --> [processing pipeline]
```

**Pros:**
- **Full BLE access on iOS and Android**
- Native performance for BLE streaming (100-200Hz achievable)
- Can run in background (with OS permissions)
- Reuses React/TypeScript expertise
- Can do local sensor fusion (quaternion estimation, step detection)
- Can cache data offline and sync later
- Push notifications for real-time feedback
- Can access phone's own IMU simultaneously

**Cons:**
- Requires app store distribution (TestFlight for iOS beta)
- Additional codebase to maintain
- App store review process for iOS
- Need to handle BLE permissions on both platforms

**Key libraries:**
- `react-native-ble-plx` (v3.5.1) -- mature, well-maintained, Expo compatible
- `react-native-ble-nitro` (v1.12.0) -- newer, higher performance via Nitro Modules, Expo compatible

**Architecture sketch:**
```typescript
// React Native companion app
import BlePlx from 'react-native-ble-plx';

const IMU_SERVICE_UUID = '...';  // Device-specific
const IMU_DATA_UUID = '...';      // Combined 9-axis characteristic

// Connect and subscribe
const device = await bleManager.connectToDevice(deviceId);
const service = await device.discoverAllServicesAndCharacteristics();
service.characteristicsForService(IMU_SERVICE_UUID).then(chars => {
  const imuChar = chars.find(c => c.uuid === IMU_DATA_UUID);
  imuChar.monitor((error, characteristic) => {
    // characteristic.value is base64-encoded binary
    const data = parseIMUPacket(characteristic.value);
    // Buffer and send via WebSocket
    ws.send(JSON.stringify(data));
  });
});
```

**Verdict: RECOMMENDED** -- best balance of capability, cross-platform support, and team expertise

---

### Option C: BLE Gateway (Raspberry Pi/ESP32) at Rink -> Server

```
[Skater's IMUs] --BLE--> [ESP32/RPi gateway at rinkside]
  --> [WiFi/MQTT/WebSocket] --> [FastAPI server]
```

**Pros:**
- Completely decouples BLE from user devices
- Can handle multiple skaters simultaneously
- No phone battery drain
- Gateway can do edge processing (sensor fusion, filtering)
- Reliable WiFi connection (vs BLE range issues)

**Cons:**
- **Additional hardware at every rink** (deployment overhead)
- Need power, WiFi, mounting at rink
- Single point of failure
- Range limitation: BLE ~10m from gateway
- Adds infrastructure cost (~$50-100 per gateway)
- Maintenance burden (firmware updates, hardware failures)

**Hardware options:**
- **ESP32-S3** ($5-10): WiFi + BLE, enough compute for sensor fusion, MicroPython/C
- **Raspberry Pi Zero W** ($15): Full Linux, Python bleak, more flexible
- **nRF52840 dongle** ($10): Dedicated BLE, excellent range

**Existing projects:**
- `nRFCloud/gateway-raspberry-pi` (9 stars): BLE gateway for RPi (archived)
- `kind3r/esp32-ble-gateway`: WiFi-to-BLE gateway on ESP32
- `particle-iot/ble-gateway-library`: Aggregates BLE peripheral data

**Verdict: VIABLE but OVER-ENGINEERED** for initial deployment. Consider for multi-skater coaching scenarios

---

### Option D: Phone as IMU (Zero Hardware) (RECOMMENDED FOR MVP)

```
[Phone in pocket] --DeviceMotion/Generic Sensor API--> [Companion App]
  --> [WebSocket] --> [FastAPI] --> [analysis]
```

**Pros:**
- **Zero hardware cost**
- Immediate deployment
- Modern phones have excellent IMUs (100Hz+, 16-bit accel/gyro)
- Sensor fusion built into OS (quaternion output available)
- No pairing complexity
- Can use Generic Sensor API (web) or native APIs (app)

**Cons:**
- Phone moves in pocket (needs strap or tight pocket)
- Placement inconsistent across sessions
- Single measurement point (can't do multi-sensor fusion across body)
- No foot-specific data (blade angle, impact)
- Gyroscope drift over long sessions

**Phone IMU capabilities (2025-2026 smartphones):**
| Sensor | Typical Spec | Sample Rate |
|--------|-------------|-------------|
| Accelerometer | 16-bit, ±16g | 100-400Hz (requestable) |
| Gyroscope | 16-bit, ±2000dps | 100-400Hz (requestable) |
| Magnetometer | 16-bit | 25-100Hz |
| Barometer | 0.01hPa | 1-25Hz |
| Fused orientation | Quaternion @ 100Hz | Hardware-level fusion |

**Sensor APIs:**

*Web (limited):*
```javascript
// DeviceMotionEvent (deprecated but widely supported)
window.addEventListener('devicemotion', (event) => {
  const accel = event.acceleration;          // {x, y, z} m/s²
  const accelGravity = event.accelerationIncludingGravity; // includes gravity
  const rotationRate = event.rotationRate;  // {alpha, beta, gamma} deg/s
  const interval = event.interval;          // ms between samples
});

// Generic Sensor API (modern, Chrome/Android only)
const accel = new Accelerometer({ frequency: 100 });
accel.addEventListener('reading', () => {
  console.log(accel.x, accel.y, accel.z);
});
accel.start();
```

*React Native (full access):*
```typescript
// react-native-sensors (cross-platform)
import { Gyroscope, Accelerometer } from 'react-native-sensors';

const gyroscope = new Gyroscope({ updateInterval: 10 }); // 100Hz
gyroscope.subscribe(({ x, y, z, timestamp }) => {
  // x, y, z in rad/s
});
```

**Verdict: BEST STARTING POINT** -- zero cost, immediate validation of IMU-based analysis pipeline

---

## 3. Data Format and Bandwidth

### 3.1 9-Axis IMU at 100Hz

**Raw data per sample:**
| Channel | Format | Bytes | Range | Resolution |
|---------|--------|-------|-------|------------|
| Accelerometer X | int16 | 2 | ±16g | ~0.5mg |
| Accelerometer Y | int16 | 2 | ±16g | ~0.5mg |
| Accelerometer Z | int16 | 2 | ±16g | ~0.5mg |
| Gyroscope X | int16 | 2 | ±2000dps | ~61mdps |
| Gyroscope Y | int16 | 2 | ±2000dps | ~61mdps |
| Gyroscope Z | int16 | 2 | ±2000dps | ~61mdps |
| Magnetometer X | int16 | 2 | ±4800uT | ~0.15uT |
| Magnetometer Y | int16 | 2 | ±4800uT | ~0.15uT |
| Magnetometer Z | int16 | 2 | ±4800uT | ~0.15uT |
| Timestamp | uint32 | 4 | ms | 1ms |
| **Total** | | **22 bytes** | | |

**At 100Hz:**
- Raw: 22 bytes/sample x 100 samples/s = **2,200 bytes/s (~17.6 kbps)**
- With JSON overhead: ~60 bytes/sample = 6,000 bytes/s (~48 kbps)
- With binary WebSocket (MessagePack/CBOR): ~24 bytes/sample = 2,400 bytes/s (~19.2 kbps)

**At 200Hz:**
- Raw: 22 x 200 = 4,400 bytes/s (~35.2 kbps)
- With JSON: ~12,000 bytes/s (~96 kbps)

### 3.2 BLE Throughput Analysis

| BLE Version | Theoretical Max | Practical Max | Notes |
|-------------|----------------|---------------|-------|
| BLE 4.0 | 305 kbps | ~100 kbps | 20-byte MTU |
| BLE 4.2 | 780 kbps | ~250 kbps | Up to 251-byte MTU |
| BLE 5.0 (2M PHY) | 1,400 kbps | ~600 kbps | Requires both ends support |

**Can BLE handle 9-axis IMU at 100Hz?** YES, easily. 2.2 KB/s is well within BLE 4.0's practical 100 kbps limit.

**Can BLE handle 9-axis IMU at 200Hz?** YES. 4.4 KB/s is still well within limits.

**Multiple sensors simultaneously?** BLE can handle ~4-6 IMU sensors at 100Hz on a single connection interval of 7.5ms. More sensors require either:
- Multiple phone connections (one phone per sensor)
- A BLE gateway that aggregates

### 3.3 Packet Loss Considerations

BLE notifications are **not acknowledged**. Typical packet loss rates:
- **Idle environment:** <0.1% (negligible)
- **Busy 2.4GHz environment** (WiFi, other BLE): 1-5%
- **Moving through interference:** 5-15%
- **Ice rink environment:** Generally low interference (large open space, few WiFi APs), expect <1%

**Buffering strategies:**
1. **Timestamp-based reconstruction:** Include monotonically increasing timestamp in each packet. Receiver can detect gaps and interpolate
2. **Sequence numbers:** Counter per packet to detect drops
3. **Circular buffer on device:** ESP32/RPi gateway can buffer 1-2 seconds of data and retransmit
4. **Application-level ACK:** For critical events (takeoff detection), request retransmission

### 3.4 Recommended Data Protocol

```typescript
// Binary format for WebSocket transmission (MessagePack)
interface IMUSample {
  t: number;    // uint32: timestamp (ms since session start)
  ax: number;   // float32: accel X (m/s²)
  ay: number;   // float32: accel Y
  az: number;   // float32: accel Z
  gx: number;   // float32: gyro X (rad/s)
  gy: number;   // float32: gyro Y
  gz: number;   // float32: gyro Z
  mx: number;   // float32: mag X (uT)
  my: number;   // float32: mag Y
  mz: number;   // float32: mag Z
}
// 40 bytes per sample with float32, or 28 bytes with quantized int16

// Batch format (send every 100ms = 10 samples at 100Hz)
interface IMUBatch {
  device_id: string;
  sensor_type: 'foot_l' | 'foot_r' | 'waist' | 'phone';
  samples: IMUSample[];  // 10 samples
  seq: number;           // batch sequence number
}
```

---

## 4. Existing Open-Source Projects

### 4.1 Web Bluetooth IMU Libraries (JS/TS)

| Project | Stars | Description | URL |
|---------|-------|-------------|-----|
| `tejaswigowda/quatro` | 0 | Web Bluetooth IMU visualizer (C++ firmware + web UI) | github.com/tejaswigowda/quatro |
| `EvanBacon/react-bluetooth` | 35 | React wrapper for Web Bluetooth API (WIP) | github.com/EvanBacon/react-bluetooth |
| `@types/web-bluetooth` | - | TypeScript definitions for Web Bluetooth | npmjs.com/package/@types/web-bluetooth |
| `thegecko/webbluetooth` | 194 | **Node.js implementation** of Web Bluetooth spec | github.com/thegecko/webbluetooth |
| `w3c/generic-sensor-demos` | - | Official W3C sensor API demos | github.com/w3c/generic-sensor-demos |
| `kenchris/sensor-polyfills` | 65 | Polyfill for Generic Sensor API | github.com/kenchris/sensor-polyfills |

### 4.2 React Native BLE Libraries

| Library | npm Version | Platform | Notes |
|---------|-------------|----------|-------|
| `react-native-ble-plx` | 3.5.1 | iOS + Android | **Most mature.** Expo compatible, active maintenance, good docs |
| `react-native-ble-nitro` | 1.12.0 | iOS + Android | **Newer, higher performance.** Built on Nitro Modules, Expo compatible |
| `react-native-ble-manager` | latest | iOS + Android | Older but stable, larger community |

### 4.3 Python BLE Libraries

| Library | Stars | Description |
|---------|-------|-------------|
| `bleak` | 2,374 | **Best choice.** Cross-platform (Linux/Windows/macOS), asyncio-based, active development |
| `bluepy` | - | Linux-only, deprecated, avoid |
| `pygatt` | - | Linux-only, wrapper around gatttool, unmaintained |
| `BleakHeart` | - | Bleak wrapper for Polar H10 (ECG + accelerometer) |

**Bleak usage for IMU:**
```python
import asyncio
from bleak import BleakClient

IMU_SERVICE = "00001826-0000-1000-8000-00805f9b34fb"
IMU_CHAR = "00002a6e-0000-1000-8000-00805f9b34fb"

async def stream_imu(device_address):
    async with BleakClient(device_address) as client:
        def notification_handler(sender, data):
            # data is bytes, parse int16 values
            ax, ay, az = struct.unpack('<hhh', data[:6])
            gx, gy, gz = struct.unpack('<hhh', data[6:12])
            # Send to FastAPI via WebSocket or process locally
            asyncio.create_task(send_to_server(ax, ay, az, gx, gy, gz))

        await client.start_notify(IMU_CHAR, notification_handler)
        await asyncio.sleep(300)  # Stream for 5 minutes
        await client.stop_notify(IMU_CHAR)
```

### 4.4 Skating-Specific BLE IMU Projects

| Project | Description | Key Finding |
|---------|-------------|-------------|
| `Mart1t1/Synergie` (AIOnIce) | Jump classification from Xsens IMU, FastAPI backend, 500 annotated jumps | **Uses Xsens + FastAPI.** Input: (400, 6) tensor, 6 IMU channels. 7 classes (6 jumps + fall). Takeoff at frame 200. Keras/TensorFlow |
| `alvisespano/sensors_2022_skate` | "A Wearable System for Jump Detection in Inline Figure Skating" (Sensors 2022) | Academic paper with public dataset. Foot-ground angle measurement via IMU |
| `dyarfaradj/Movesense-Bluetooth-Sensor` | React Native app reading Movesense IMU data (accel, gyro, mag, HR) | **Proof of concept for Movesense + React Native.** 6 stars |

### 4.5 Relevant IMU MoCap Projects

| Project | Stars | Description |
|---------|-------|-------------|
| `xioTechnologies/IMU-Mocap` | 15 | Open-source IMU motion capture (C#, Python pip: `imumocap`) |
| `CheCap/CheCap` | - | Cheap IMU MoCap using multiple IMUs |
| `fsmeraldi/bleakheart` | - | Async BLE heart monitor with accelerometer data (built on bleak) |

---

## 5. Phone-as-IMU: Detailed Analysis

### 5.1 Modern Phone IMU Quality (2025-2026)

Modern smartphones have **excellent IMUs** that rival dedicated sensor modules:

| Phone | Accelerometer | Gyroscope | Notes |
|-------|--------------|-----------|-------|
| iPhone 15 Pro | BMI323 (Bosch) | BMI323 | 16-bit, 400Hz, hardware sensor fusion |
| Samsung S24 | ICM-42688 (InvenSense) | ICM-42688 | 16-bit, 400Hz, 6-axis |
| Pixel 9 | ICM-40607 | ICM-40607 | 16-bit, 200Hz |
| Generic mid-range | Various | Various | 50-100Hz typically |

**Key advantage:** Phones have **hardware-level sensor fusion** (ARM CMSIS-DSP or vendor-specific). You get quaternions directly from the OS, not raw data you need to fuse yourself.

### 5.2 Can a Phone in a Pocket Provide Useful Skating Data?

**Yes, for specific metrics:**

| Metric | Feasibility | Notes |
|--------|------------|-------|
| Jump detection (takeoff/landing) | **HIGH** | Vertical acceleration spike is unmistakable |
| Jump height (flight time) | **HIGH** | Accelerometer double-integration with drift correction |
| Rotation count | **MEDIUM** | Gyroscope integration, drift over 1-2s rotations |
| Edge type (inside/outside) | **LOW** | Phone at waist can't detect foot angle |
| Blade impact force | **LOW** | Phone too far from blade |
| Body lean angle | **HIGH** | Phone pitch/roll directly measures body lean |
| Angular velocity | **HIGH** | Gyroscope measures spin rate directly |
| Step count / cadence | **MEDIUM** | Detectable from waist acceleration patterns |

**Best phone placement for skating:**
1. **Waist belt (tight):** Best overall -- captures body lean, rotation, vertical oscillation
2. **Thigh strap:** Good for jump detection, closer to leg dynamics
3. **Chest harness:** Best for upper body rotation, but uncomfortable for skating
4. **Pocket:** Least reliable (moves around), but zero-equipment

### 5.3 Sensor Fusion APIs

**Web (limited):**
```javascript
// DeviceOrientationEvent (absolute orientation, includes magnetometer fusion)
window.addEventListener('deviceorientation', (event) => {
  const alpha = event.alpha; // compass heading (0-360)
  const beta = event.beta;   // front-back tilt (-180 to 180)
  const gamma = event.gamma; // left-right tilt (-90 to 90)
});

// DeviceMotionEvent (raw + gravity-compensated)
window.addEventListener('devicemotion', (event) => {
  const accel = event.acceleration;           // linear acceleration (gravity removed)
  const accelG = event.accelerationIncludingGravity; // with gravity
  const rate = event.rotationRate;            // deg/s
});
```

**React Native (full access):**
```typescript
import { Gyroscope, Accelerometer, Magnetometer } from 'react-native-sensors';

// Accelerometer at 100Hz
const accel = new Accelerometer({ updateInterval: 10 });
accel.subscribe(({ x, y, z, timestamp }) => {
  // x, y, z in m/s²
});

// Gyroscope at 100Hz
const gyro = new Gyroscope({ updateInterval: 10 });
gyro.subscribe(({ x, y, z, timestamp }) => {
  // x, y, z in rad/s
});

// For fused orientation, use react-native-sensors or expo-sensors
// which provide quaternion output on platforms that support it
```

### 5.4 Accuracy vs Dedicated IMU

| Metric | Phone (waist) | Dedicated IMU (foot) | Difference |
|--------|--------------|---------------------|------------|
| Jump height | ±5cm | ±2cm | Phone: double integration drift |
| Rotation count | ±0.25 rev | ±0.1 rev | Phone: gyroscope drift during spin |
| Takeoff time | ±10ms | ±5ms | Phone: sufficient for both |
| Landing quality | Low accuracy | High accuracy | Foot-mounted needed for blade angle |
| Body lean | ±2° | N/A | Phone is actually better positioned |

---

## 6. Recommended Architecture

### Phase 1: Phone-as-IMU MVP (Week 1-2)

```
[Phone in waist belt]
  --> [React Native companion app]
    --> DeviceMotion API (accel, gyro, orientation)
    --> WebSocket to FastAPI
  --> [FastAPI processing]
    --> Jump detection (accelerometer peaks)
    --> Rotation counting (gyroscope integration)
    --> Body lean analysis (orientation quaternion)
    --> Real-time feedback via WebSocket
  --> [Existing video analysis pipeline]
    --> Correlate IMU events with video frames
```

**Why start here:**
- Zero hardware cost
- Validates IMU data processing pipeline
- Coaches can try immediately
- Builds WebSocket infrastructure needed for all options

### Phase 2: BLE IMU Integration (Month 2-3)

```
[Phone in pocket] + [BLE IMUs on boots]
  --> [React Native companion app]
    --> Phone's own IMU (waist reference)
    --> react-native-ble-plx connects to boot IMUs
    --> Sensor fusion (phone + boot data)
    --> WebSocket batched data to FastAPI
  --> [FastAPI processing]
    --> Multi-sensor fusion (Madgwick/Mahony filter)
    --> Blade angle detection (foot IMU)
    --> Edge type classification
    --> Landing quality assessment
    --> Correlation with video pose estimation
  --> [Visualization]
    --> Overlay IMU data on video
    --> Compare IMU metrics with video-based metrics
```

### Phase 3: Rink Gateway (Optional, Multi-Skater)

```
[Multiple skaters with BLE IMUs]
  --> [ESP32-S3 gateway at rinkside]
    --> BLE central (connects to all skater IMUs)
    --> Local sensor fusion
    --> WiFi/MQTT to cloud server
  --> [FastAPI server]
    --> Multi-skater tracking
    --> Session recording
    --> Real-time leaderboard
```

---

## 7. Integration with Existing Pipeline

The IMU data complements the existing video-based analysis:

| Video-Based (current) | IMU-Based (new) | Combined |
|-----------------------|-----------------|----------|
| RTMPose skeleton | Jump height (CoM) | Validate video estimates with ground truth |
| Phase detection (CoM) | Takeoff/landing timing | Higher precision phase boundaries |
| Body lean (pose) | Body lean (orientation) | Cross-validate |
| Blade angle (estimation) | Blade angle (foot IMU) | Direct measurement replaces estimation |
| Rotation speed (visual) | Rotation speed (gyro) | Ground truth calibration |

**Key integration point:** Time synchronization. IMU timestamps must align with video frames. Solutions:
1. **Manual sync:** Clap/tap at start of recording (visible in video + IMU spike)
2. **Audio sync:** Detect clap sound in video audio and IMU accelerometer
3. **NTP-based:** If both devices have internet, sync to NTP

---

## 8. Cost Estimate

| Component | Cost | Quantity | Total |
|-----------|------|----------|-------|
| Phase 1: Phone-as-IMU | $0 | - | $0 |
| Phase 2: Movesense sensors | $150-200 each | 2 (boots) | $300-400 |
| Phase 2: Belt mount | $20 | 1 | $20 |
| Phase 2: React Native dev | Existing team | - | $0 |
| Phase 3: ESP32-S3 gateway | $10 | 1 | $10 |
| Phase 3: RPi + power | $50 | 1 | $50 |
| **Total Phase 2** | | | **$320-420** |
| **Total Phase 3** | | | **$330-430** |

---

## 9. Key Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| iOS BLE permission denied | Low | High | Clear UX, guide user through Settings |
| BLE packet loss during jumps | Medium | Medium | Buffer + timestamp reconstruction |
| IMU drift during long spins | Medium | Medium | Complementary filter (accel correction) |
| Phone IMU sample rate throttling | Medium | Medium | Use native APIs (React Native), not web APIs |
| Time sync drift between video and IMU | Medium | High | Audio clap sync, periodic resync |
| Battery drain on phone | Low | Low | BLE is low power (~10mA), phone lasts hours |

---

## 10. References

- AIOnIce / Synergie: `github.com/Mart1t1/Synergie` -- Xsens IMU + FastAPI for skating jump classification
- Movesense React Native: `github.com/dyarfaradj/Movesense-Bluetooth-Sensor` -- React Native reading Movesense IMU
- Quatro IMU Visualizer: `github.com/tejaswigowda/quatro` -- Web Bluetooth IMU visualization
- Bleak Python BLE: `github.com/hbldh/bleak` (2,374 stars) -- Cross-platform Python BLE client
- react-native-ble-plx: `npmjs.com/package/react-native-ble-plx` (v3.5.1) -- Mature React Native BLE
- Web Bluetooth API: `developer.mozilla.org/en-US/docs/Web/API/Web_Bluetooth_API`
- Generic Sensor API: `github.com/w3c/sensors`
- IMU-MoCap: `github.com/xioTechnologies/IMU-Mocap` -- Open-source IMU motion capture
- Flutter BLE Broadcaster: `github.com/IoT-gamer/flutter_ble_sensor_broadcaster` -- Phone as BLE peripheral
- Sensors 2022 paper: "A Wearable System for Jump Detection in Inline Figure Skating"
