/*
 * Modular Direction Sensor - All Methods (v2.0)
 * 
 * Changes from v1:
 *   - Real ToF support using SparkFun VL53L5CX library
 *   - Calibration mode for coordinate mapping
 *   - Direction flip support
 * 
 * Radar Methods:
 *   1. TRIPWIRE - Detect crossing direction from density analysis
 *   2. CORRELATION - Camera tells count, radar estimates from duration
 * 
 * ToF Methods:
 *   1. SINGLE_TRIPWIRE - Whole grid as one presence detector
 *   2. DUAL_ZONE - Inside/outside zones, detect sequence
 * 
 * Serial Commands:
 *   radar tripwire     - Switch radar to tripwire mode
 *   radar correlation  - Switch radar to correlation mode
 *   tof single         - Switch ToF to single tripwire
 *   tof zones          - Switch ToF to dual zone mode
 *   count <n>          - Tell radar to expect n people (for correlation)
 *   flip on/off        - Flip IN/OUT direction
 *   calibrate          - Enter calibration mode (shows raw sensor data)
 *   status             - Show current modes and state
 *   raw on/off         - Toggle raw data output for debugging
 * 
 * Install Library:
 *   Arduino IDE -> Sketch -> Include Library -> Manage Libraries
 *   Search: "SparkFun VL53L5CX" -> Install
 */

#include <Wire.h>
#include <SparkFun_VL53L5CX_Library.h>

// ============================================================================
// PIN DEFINITIONS
// ============================================================================

// Radar (HLK-LD6001A) - UART2
#define RADAR_RX_PIN    16
#define RADAR_TX_PIN    17
#define RADAR_RST_PIN   4
#define RADAR_BAUD      921600

// ToF (VL53L5CX) - I2C
#define TOF_SDA_PIN     21
#define TOF_SCL_PIN     22
#define TOF_INT_PIN     19
#define TOF_LPN_PIN     18

// ============================================================================
// MODE DEFINITIONS
// ============================================================================

enum RadarMode {
  RADAR_OFF,
  RADAR_TRIPWIRE,
  RADAR_CORRELATION
};

enum ToFMode {
  TOF_OFF,
  TOF_SINGLE_TRIPWIRE,
  TOF_DUAL_ZONE
};

// Current modes (runtime changeable)
RadarMode radarMode = RADAR_TRIPWIRE;
ToFMode tofMode = TOF_SINGLE_TRIPWIRE;
bool rawOutputEnabled = false;
bool calibrateMode = false;
bool flipDirection = false;  // When true, swap IN/OUT

// ============================================================================
// RADAR: DATA STRUCTURES (DEBUG=1 Binary Format)
// ============================================================================

#define RADAR_BUFFER_SIZE 512

struct RadarAnalysis {
  uint8_t buffer[RADAR_BUFFER_SIZE];
  uint16_t bufferIndex;
  
  // Analysis results
  int firstHalfCount;   // 0x80 bytes in first half
  int secondHalfCount;  // 0x80 bytes in second half
  int totalCount;       // Total 0x80 bytes
  float density;        // Percentage of 0x80 bytes
  float ratio;          // firstHalf / secondHalf (>1 = approaching)
  
  // Smoothed values
  float smoothedRatio;
  float smoothedDensity;
  
  unsigned long timestamp;
};

RadarAnalysis radarData;

// ============================================================================
// RADAR: TRIPWIRE STATE
// ============================================================================

struct TripwireState {
  float minDensityPercent = 10.0;
  float minRatioDiff = 0.15;
  float smoothingAlpha = 0.3;
  
  int8_t currentDirection;  // 0=none, 1=approaching, 2=receding
  float confidence;
  
  unsigned long lastTriggerTime = 0;
  unsigned long debounceMs = 300;
  
  bool motionActive = false;
  unsigned long motionStartTime = 0;
  int8_t motionDirection = 0;
} tripwire;

// ============================================================================
// RADAR: CORRELATION STATE
// ============================================================================

struct CorrelationState {
  uint8_t expectedCount;
  bool cameraCountReceived;
  unsigned long countReceivedTime;
  
  uint8_t motionEventsDetected;
  unsigned long lastMotionEnd;
  uint16_t totalMotionDuration;
  
  float durationPerPerson = 400;
} correlation;

// ============================================================================
// TOF: DATA STRUCTURES (Using SparkFun Library)
// ============================================================================

SparkFun_VL53L5CX tofSensor;
VL53L5CX_ResultsData tofResults;
bool tofInitialized = false;

struct ToFFrame {
  int16_t distance[64];   // Distance in mm (8x8 grid), -1 = invalid
  uint8_t status[64];     // Measurement status per zone
  bool valid;
  unsigned long timestamp;
};

ToFFrame currentToFFrame;

// ============================================================================
// TOF: TRIPWIRE STATE
// ============================================================================

struct ToFTripwireState {
  uint16_t clearDistance = 2000;   // mm - nothing closer = clear
  uint16_t minValidDistance = 50;  // mm - ignore closer than this
  
  bool triggered;
  bool previousTriggered;
  unsigned long triggerStartTime;
  unsigned long triggerDuration;
  uint8_t triggeredZoneCount;
} tofTripwire;

// ============================================================================
// TOF: DUAL ZONE STATE
// ============================================================================

struct ToFDualZoneState {
  // Zone definitions (column-based)
  // Columns 0-3 = OUTSIDE (away from door), Columns 4-7 = INSIDE (toward door)
  
  bool outsideTriggered;
  bool insideTriggered;
  unsigned long outsideTime;
  unsigned long insideTime;
  
  uint8_t outsideColumnCount;
  uint8_t insideColumnCount;
  
  int8_t lastDirection;  // 1=entry, 2=exit, 0=none
  float confidence;
} tofZones;

// ============================================================================
// OUTPUT EVENTS
// ============================================================================

#define EVENT_NONE              0
#define EVENT_RADAR_ENTRY       1
#define EVENT_RADAR_EXIT        2
#define EVENT_RADAR_MOTION      3
#define EVENT_TOF_TRIGGERED     4
#define EVENT_TOF_CLEARED       5
#define EVENT_TOF_ENTRY         6
#define EVENT_TOF_EXIT          7
#define EVENT_CROSSING_CONFIRMED 8

void sendEvent(uint8_t eventType, float confidence, const char* source, const char* extra = "") {
  // Apply direction flip if enabled
  uint8_t finalEvent = eventType;
  if (flipDirection) {
    if (eventType == EVENT_RADAR_ENTRY) finalEvent = EVENT_RADAR_EXIT;
    else if (eventType == EVENT_RADAR_EXIT) finalEvent = EVENT_RADAR_ENTRY;
    else if (eventType == EVENT_TOF_ENTRY) finalEvent = EVENT_TOF_EXIT;
    else if (eventType == EVENT_TOF_EXIT) finalEvent = EVENT_TOF_ENTRY;
  }
  
  Serial.printf("EVT,%d,%.2f,%s,%s\n", finalEvent, confidence, source, extra);
}

// ============================================================================
// RADAR SERIAL PARSING
// ============================================================================

HardwareSerial RadarSerial(2);

#define RADAR_ANALYSIS_INTERVAL_MS 150

unsigned long lastRadarAnalysis = 0;

void radarBufferByte(uint8_t byte) {
  if (byte == 0x00 || byte == 0x80) {
    radarData.buffer[radarData.bufferIndex++] = byte;
    if (radarData.bufferIndex >= RADAR_BUFFER_SIZE) {
      radarData.bufferIndex = 0;
    }
  }
}

void analyzeRadarBuffer() {
  radarData.firstHalfCount = 0;
  radarData.secondHalfCount = 0;
  
  int halfPoint = RADAR_BUFFER_SIZE / 2;
  
  for (int i = 0; i < RADAR_BUFFER_SIZE; i++) {
    if (radarData.buffer[i] == 0x80) {
      if (i < halfPoint) {
        radarData.firstHalfCount++;
      } else {
        radarData.secondHalfCount++;
      }
    }
  }
  
  radarData.totalCount = radarData.firstHalfCount + radarData.secondHalfCount;
  radarData.density = (float)radarData.totalCount / RADAR_BUFFER_SIZE * 100.0;
  
  if (radarData.secondHalfCount > 0) {
    radarData.ratio = (float)radarData.firstHalfCount / radarData.secondHalfCount;
  } else if (radarData.firstHalfCount > 0) {
    radarData.ratio = 10.0;
  } else {
    radarData.ratio = 1.0;
  }
  
  // Smooth values
  radarData.smoothedDensity = radarData.smoothedDensity * (1 - tripwire.smoothingAlpha) + 
                              radarData.density * tripwire.smoothingAlpha;
  radarData.smoothedRatio = radarData.smoothedRatio * (1 - tripwire.smoothingAlpha) + 
                            radarData.ratio * tripwire.smoothingAlpha;
  
  radarData.timestamp = millis();
}

// ============================================================================
// RADAR: TRIPWIRE PROCESSING
// ============================================================================

void processRadarTripwire() {
  unsigned long now = millis();
  
  // Output raw data in calibrate mode
  if (calibrateMode) {
    Serial.printf("CAL,RADAR,density=%.1f,ratio=%.2f,first=%d,second=%d\n",
                  radarData.smoothedDensity, radarData.smoothedRatio,
                  radarData.firstHalfCount, radarData.secondHalfCount);
  }
  
  if (rawOutputEnabled) {
    Serial.printf("RAW,RADAR,%.1f,%.2f\n", radarData.smoothedDensity, radarData.smoothedRatio);
  }
  
  // Check if motion detected
  bool hasMotion = radarData.smoothedDensity >= tripwire.minDensityPercent;
  
  // Determine direction from ratio
  int8_t direction = 0;
  float conf = 0.0;
  
  if (hasMotion) {
    float ratioDiff = radarData.smoothedRatio - 1.0;
    
    if (ratioDiff > tripwire.minRatioDiff) {
      direction = 1;  // Approaching (more in first half = newer data)
      conf = min(1.0f, ratioDiff / 0.5f);
    } else if (ratioDiff < -tripwire.minRatioDiff) {
      direction = 2;  // Receding
      conf = min(1.0f, -ratioDiff / 0.5f);
    }
  }
  
  // Motion state machine
  if (hasMotion && !tripwire.motionActive) {
    // Motion started
    tripwire.motionActive = true;
    tripwire.motionStartTime = now;
    tripwire.motionDirection = direction;
    sendEvent(EVENT_RADAR_MOTION, conf, "radar_tripwire", "start");
  }
  else if (!hasMotion && tripwire.motionActive) {
    // Motion ended - determine final direction
    tripwire.motionActive = false;
    unsigned long duration = now - tripwire.motionStartTime;
    
    if (duration > 100 && now - tripwire.lastTriggerTime > tripwire.debounceMs) {
      tripwire.lastTriggerTime = now;
      
      if (tripwire.motionDirection == 1) {
        sendEvent(EVENT_RADAR_ENTRY, tripwire.confidence, "radar_tripwire", "approaching");
      } else if (tripwire.motionDirection == 2) {
        sendEvent(EVENT_RADAR_EXIT, tripwire.confidence, "radar_tripwire", "receding");
      }
    }
  }
  else if (hasMotion && tripwire.motionActive) {
    // Update direction during motion (use strongest signal)
    if (conf > tripwire.confidence) {
      tripwire.motionDirection = direction;
      tripwire.confidence = conf;
    }
  }
  
  tripwire.currentDirection = direction;
}

// ============================================================================
// RADAR: CORRELATION PROCESSING
// ============================================================================

void processRadarCorrelation() {
  unsigned long now = millis();
  
  if (!correlation.cameraCountReceived) return;
  
  // Timeout after 5 seconds
  if (now - correlation.countReceivedTime > 5000) {
    correlation.cameraCountReceived = false;
    return;
  }
  
  bool hasMotion = radarData.smoothedDensity >= tripwire.minDensityPercent;
  
  // Track motion events
  static bool wasMotion = false;
  static unsigned long motionStart = 0;
  
  if (hasMotion && !wasMotion) {
    motionStart = now;
    wasMotion = true;
  }
  else if (!hasMotion && wasMotion) {
    wasMotion = false;
    correlation.motionEventsDetected++;
    correlation.totalMotionDuration += (now - motionStart);
    correlation.lastMotionEnd = now;
  }
  
  // Check if crossing complete (500ms of no motion after activity)
  if (correlation.motionEventsDetected > 0 && 
      !hasMotion && 
      now - correlation.lastMotionEnd > 500) {
    
    // Estimate count from duration
    int estimatedCount = max(1, (int)(correlation.totalMotionDuration / correlation.durationPerPerson + 0.5));
    
    // Confidence based on match
    float conf = 1.0 - (abs(estimatedCount - correlation.expectedCount) * 0.3);
    conf = max(0.3f, conf);
    
    char extra[32];
    sprintf(extra, "exp=%d,est=%d,dur=%d", 
            correlation.expectedCount, estimatedCount, correlation.totalMotionDuration);
    
    sendEvent(EVENT_CROSSING_CONFIRMED, conf, "radar_correlation", extra);
    
    // Reset
    correlation.cameraCountReceived = false;
    correlation.motionEventsDetected = 0;
    correlation.totalMotionDuration = 0;
  }
}

// ============================================================================
// TOF INITIALIZATION (SparkFun Library)
// ============================================================================

void tofInit() {
  Wire.begin(TOF_SDA_PIN, TOF_SCL_PIN);
  Wire.setClock(400000);  // 400kHz I2C
  
  // LPN pin controls sensor enable
  pinMode(TOF_LPN_PIN, OUTPUT);
  digitalWrite(TOF_LPN_PIN, HIGH);  // Enable sensor
  delay(10);
  
  Serial.println("[ToF] Initializing VL53L5CX...");
  
  if (tofSensor.begin() == false) {
    Serial.println("[ToF] Sensor not found. Check wiring.");
    tofInitialized = false;
    return;
  }
  
  // Configure for 8x8 resolution, 15Hz
  tofSensor.setResolution(8*8);
  tofSensor.setRangingFrequency(15);
  tofSensor.startRanging();
  
  tofInitialized = true;
  Serial.println("[ToF] Initialized: 8x8 @ 15Hz");
}

// ============================================================================
// TOF DATA READING
// ============================================================================

bool tofReadData() {
  if (!tofInitialized) return false;
  
  if (tofSensor.isDataReady()) {
    if (tofSensor.getRangingData(&tofResults)) {
      currentToFFrame.timestamp = millis();
      currentToFFrame.valid = true;
      
      // Copy distance data (8x8 = 64 zones)
      for (int i = 0; i < 64; i++) {
        // Check status - only use valid measurements
        uint8_t status = tofResults.target_status[i];
        if (status == 5 || status == 6 || status == 9) {
          // Valid measurement
          currentToFFrame.distance[i] = tofResults.distance_mm[i];
        } else {
          currentToFFrame.distance[i] = -1;  // Invalid
        }
        currentToFFrame.status[i] = status;
      }
      return true;
    }
  }
  return false;
}

// ============================================================================
// TOF: SINGLE TRIPWIRE PROCESSING
// ============================================================================

void processToFSingleTripwire() {
  if (!tofReadData()) return;
  
  unsigned long now = millis();
  int triggeredCount = 0;
  int minDistance = 9999;
  
  // Check all zones
  for (int i = 0; i < 64; i++) {
    int16_t dist = currentToFFrame.distance[i];
    if (dist > tofTripwire.minValidDistance && dist < tofTripwire.clearDistance) {
      triggeredCount++;
      if (dist < minDistance) minDistance = dist;
    }
  }
  
  bool triggered = triggeredCount > 2;  // At least 3 zones
  
  // Output in calibrate mode
  if (calibrateMode) {
    // Find which columns are triggered
    int colCounts[8] = {0};
    for (int col = 0; col < 8; col++) {
      for (int row = 0; row < 8; row++) {
        int idx = row * 8 + col;
        int16_t dist = currentToFFrame.distance[idx];
        if (dist > tofTripwire.minValidDistance && dist < tofTripwire.clearDistance) {
          colCounts[col]++;
        }
      }
    }
    Serial.printf("CAL,TOF,triggered=%d,min=%d,cols=[%d,%d,%d,%d,%d,%d,%d,%d]\n",
                  triggeredCount, minDistance,
                  colCounts[0], colCounts[1], colCounts[2], colCounts[3],
                  colCounts[4], colCounts[5], colCounts[6], colCounts[7]);
  }
  
  if (rawOutputEnabled && triggered) {
    Serial.printf("RAW,TOF_SINGLE,%d,%d\n", triggeredCount, minDistance);
  }
  
  // State change detection
  if (triggered && !tofTripwire.previousTriggered) {
    tofTripwire.triggered = true;
    tofTripwire.triggerStartTime = now;
    tofTripwire.triggeredZoneCount = triggeredCount;
    sendEvent(EVENT_TOF_TRIGGERED, 1.0, "tof_tripwire", String(triggeredCount).c_str());
  }
  else if (!triggered && tofTripwire.previousTriggered) {
    tofTripwire.triggered = false;
    tofTripwire.triggerDuration = now - tofTripwire.triggerStartTime;
    sendEvent(EVENT_TOF_CLEARED, 1.0, "tof_tripwire", String(tofTripwire.triggerDuration).c_str());
  }
  
  tofTripwire.previousTriggered = triggered;
}

// ============================================================================
// TOF: DUAL ZONE PROCESSING
// ============================================================================

void processToFDualZone() {
  if (!tofReadData()) return;
  
  unsigned long now = millis();
  
  // Count triggers in outside columns (0-3) and inside columns (4-7)
  int outsideCount = 0;
  int insideCount = 0;
  uint8_t outsideCols = 0;
  uint8_t insideCols = 0;
  
  for (int col = 0; col < 8; col++) {
    int colTriggers = 0;
    for (int row = 0; row < 8; row++) {
      int idx = row * 8 + col;
      int16_t dist = currentToFFrame.distance[idx];
      if (dist > tofTripwire.minValidDistance && dist < tofTripwire.clearDistance) {
        colTriggers++;
      }
    }
    
    if (colTriggers >= 2) {  // At least 2 zones in column
      if (col < 4) {
        outsideCount += colTriggers;
        outsideCols++;
      } else {
        insideCount += colTriggers;
        insideCols++;
      }
    }
  }
  
  bool outsideActive = outsideCols >= 1;
  bool insideActive = insideCols >= 1;
  
  // Output in calibrate mode
  if (calibrateMode) {
    Serial.printf("CAL,TOF_ZONE,outside=%d(cols=%d),inside=%d(cols=%d)\n",
                  outsideCount, outsideCols, insideCount, insideCols);
  }
  
  if (rawOutputEnabled && (outsideActive || insideActive)) {
    Serial.printf("RAW,TOF_ZONE,outside=%d(%d),inside=%d(%d)\n", 
                  outsideActive, outsideCols, insideActive, insideCols);
  }
  
  // Track trigger times
  if (outsideActive && !tofZones.outsideTriggered) {
    tofZones.outsideTriggered = true;
    tofZones.outsideTime = now;
    tofZones.outsideColumnCount = outsideCols;
  }
  if (insideActive && !tofZones.insideTriggered) {
    tofZones.insideTriggered = true;
    tofZones.insideTime = now;
    tofZones.insideColumnCount = insideCols;
  }
  
  // Clear triggers when zone clears
  if (!outsideActive && tofZones.outsideTriggered) {
    // Only clear after short delay to handle momentary gaps
    if (now - tofZones.outsideTime > 200) {
      tofZones.outsideTriggered = false;
    }
  }
  if (!insideActive && tofZones.insideTriggered) {
    if (now - tofZones.insideTime > 200) {
      tofZones.insideTriggered = false;
    }
  }
  
  // Determine direction when both have been triggered
  if (tofZones.outsideTriggered && tofZones.insideTriggered) {
    int totalColumns = tofZones.outsideColumnCount + tofZones.insideColumnCount;
    float widthPenalty = totalColumns > 4 ? 0.6 : 1.0;  // Wide = grouped entry
    
    unsigned long timeDiff;
    if (tofZones.outsideTime < tofZones.insideTime) {
      // Outside first -> Entry (walking toward inside)
      timeDiff = tofZones.insideTime - tofZones.outsideTime;
      tofZones.lastDirection = 1;
      tofZones.confidence = widthPenalty * max(0.5f, 1.0f - timeDiff / 1000.0f);
      sendEvent(EVENT_TOF_ENTRY, tofZones.confidence, "tof_zones",
                totalColumns > 4 ? "grouped" : "single");
    } else {
      // Inside first -> Exit (walking toward outside)
      timeDiff = tofZones.outsideTime - tofZones.insideTime;
      tofZones.lastDirection = 2;
      tofZones.confidence = widthPenalty * max(0.5f, 1.0f - timeDiff / 1000.0f);
      sendEvent(EVENT_TOF_EXIT, tofZones.confidence, "tof_zones",
                totalColumns > 4 ? "grouped" : "single");
    }
    
    // Reset for next detection
    tofZones.outsideTriggered = false;
    tofZones.insideTriggered = false;
  }
  
  // Timeout stale single-zone triggers
  if (tofZones.outsideTriggered && !tofZones.insideTriggered && 
      (now - tofZones.outsideTime > 2000)) {
    tofZones.outsideTriggered = false;
  }
  if (tofZones.insideTriggered && !tofZones.outsideTriggered && 
      (now - tofZones.insideTime > 2000)) {
    tofZones.insideTriggered = false;
  }
}

// ============================================================================
// SERIAL COMMAND HANDLER
// ============================================================================

void processSerialCommands() {
  if (!Serial.available()) return;
  
  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  cmd.toLowerCase();
  
  if (cmd == "status") {
    Serial.println("\n=== STATUS ===");
    Serial.printf("Radar mode: %s\n", 
                  radarMode == RADAR_OFF ? "OFF" :
                  radarMode == RADAR_TRIPWIRE ? "TRIPWIRE" : "CORRELATION");
    Serial.printf("ToF mode: %s\n",
                  tofMode == TOF_OFF ? "OFF" :
                  tofMode == TOF_SINGLE_TRIPWIRE ? "SINGLE_TRIPWIRE" : "DUAL_ZONE");
    Serial.printf("ToF initialized: %s\n", tofInitialized ? "YES" : "NO");
    Serial.printf("Direction flip: %s\n", flipDirection ? "ON (IN<->OUT swapped)" : "OFF");
    Serial.printf("Calibrate mode: %s\n", calibrateMode ? "ON" : "OFF");
    Serial.printf("Raw output: %s\n", rawOutputEnabled ? "ON" : "OFF");
    Serial.printf("Radar density: %.1f%%\n", radarData.smoothedDensity);
    Serial.printf("Radar ratio: %.2f (>1=approaching, <1=receding)\n", radarData.smoothedRatio);
    Serial.println("==============\n");
  }
  else if (cmd == "radar tripwire") {
    radarMode = RADAR_TRIPWIRE;
    Serial.println("OK: Radar = TRIPWIRE");
  }
  else if (cmd == "radar correlation") {
    radarMode = RADAR_CORRELATION;
    Serial.println("OK: Radar = CORRELATION");
  }
  else if (cmd == "radar off") {
    radarMode = RADAR_OFF;
    Serial.println("OK: Radar = OFF");
  }
  else if (cmd == "tof single") {
    tofMode = TOF_SINGLE_TRIPWIRE;
    Serial.println("OK: ToF = SINGLE_TRIPWIRE");
  }
  else if (cmd == "tof zones") {
    tofMode = TOF_DUAL_ZONE;
    Serial.println("OK: ToF = DUAL_ZONE");
  }
  else if (cmd == "tof off") {
    tofMode = TOF_OFF;
    Serial.println("OK: ToF = OFF");
  }
  else if (cmd.startsWith("count ")) {
    int count = cmd.substring(6).toInt();
    if (count > 0 && count <= 10) {
      correlation.expectedCount = count;
      correlation.cameraCountReceived = true;
      correlation.countReceivedTime = millis();
      correlation.motionEventsDetected = 0;
      correlation.totalMotionDuration = 0;
      Serial.printf("OK: Expecting %d people\n", count);
    }
  }
  else if (cmd == "flip on") {
    flipDirection = true;
    Serial.println("OK: Direction FLIPPED (IN<->OUT swapped)");
  }
  else if (cmd == "flip off") {
    flipDirection = false;
    Serial.println("OK: Direction NORMAL");
  }
  else if (cmd == "calibrate" || cmd == "calibrate on") {
    calibrateMode = true;
    Serial.println("OK: Calibrate mode ON");
    Serial.println("Walk through sensor field to see coordinate data:");
    Serial.println("  CAL,RADAR,density=X,ratio=Y,...");
    Serial.println("  CAL,TOF,triggered=N,min=D,cols=[...]");
    Serial.println("  CAL,TOF_ZONE,outside=N,inside=M,...");
    Serial.println("Type 'calibrate off' to stop");
  }
  else if (cmd == "calibrate off") {
    calibrateMode = false;
    Serial.println("OK: Calibrate mode OFF");
  }
  else if (cmd == "raw on") {
    rawOutputEnabled = true;
    Serial.println("OK: Raw output ON");
  }
  else if (cmd == "raw off") {
    rawOutputEnabled = false;
    Serial.println("OK: Raw output OFF");
  }
  else if (cmd.startsWith("at+")) {
    RadarSerial.print(cmd + "\n");
    delay(100);
    while (RadarSerial.available()) {
      Serial.print((char)RadarSerial.read());
    }
  }
  else if (cmd == "help") {
    Serial.println("\n=== COMMANDS ===");
    Serial.println("radar tripwire    - Binary density direction detection");
    Serial.println("radar correlation - Camera-guided timing");
    Serial.println("radar off         - Disable radar");
    Serial.println("tof single        - ToF as simple tripwire");
    Serial.println("tof zones         - ToF with inside/outside zones");
    Serial.println("tof off           - Disable ToF");
    Serial.println("count <n>         - Expect n people (correlation mode)");
    Serial.println("flip on/off       - Swap IN<->OUT direction");
    Serial.println("calibrate         - Show raw coordinate data");
    Serial.println("raw on/off        - Toggle raw output");
    Serial.println("status            - Show current state");
    Serial.println("at+<cmd>          - Pass to radar");
    Serial.println("================\n");
  }
}

// ============================================================================
// RADAR INITIALIZATION
// ============================================================================

void radarInit() {
  RadarSerial.begin(RADAR_BAUD, SERIAL_8N1, RADAR_RX_PIN, RADAR_TX_PIN);
  
  pinMode(RADAR_RST_PIN, OUTPUT);
  digitalWrite(RADAR_RST_PIN, LOW);
  delay(100);
  digitalWrite(RADAR_RST_PIN, HIGH);
  delay(500);
  
  RadarSerial.print("AT+DEBUG=1\n");
  delay(100);
  
  while (RadarSerial.available()) {
    RadarSerial.read();
  }
  
  Serial.println("[Radar] Initialized (DEBUG=1)");
}

// ============================================================================
// MAIN SETUP AND LOOP
// ============================================================================

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n==========================================");
  Serial.println("  Direction Sensor v2.0 (SparkFun ToF)");
  Serial.println("==========================================");
  Serial.println("Type 'help' for commands\n");
  
  memset(radarData.buffer, 0, RADAR_BUFFER_SIZE);
  radarData.bufferIndex = 0;
  radarData.smoothedRatio = 1.0;
  radarData.smoothedDensity = 0.0;
  
  radarInit();
  tofInit();
  
  Serial.println("\nReady:");
  Serial.printf("  Radar: %s\n", radarMode == RADAR_TRIPWIRE ? "TRIPWIRE" : "CORRELATION");
  Serial.printf("  ToF: %s (%s)\n", 
                tofMode == TOF_SINGLE_TRIPWIRE ? "SINGLE_TRIPWIRE" : "DUAL_ZONE",
                tofInitialized ? "OK" : "NOT FOUND");
  Serial.printf("  Flip: %s\n", flipDirection ? "ON" : "OFF");
  Serial.println("");
}

void loop() {
  processSerialCommands();
  
  // Buffer radar data
  while (RadarSerial.available()) {
    radarBufferByte(RadarSerial.read());
  }
  
  // Analyze radar
  unsigned long now = millis();
  if (radarMode != RADAR_OFF && (now - lastRadarAnalysis >= RADAR_ANALYSIS_INTERVAL_MS)) {
    lastRadarAnalysis = now;
    analyzeRadarBuffer();
    
    if (radarMode == RADAR_TRIPWIRE) {
      processRadarTripwire();
    } else if (radarMode == RADAR_CORRELATION) {
      processRadarCorrelation();
    }
  }
  
  // Process ToF
  if (tofMode == TOF_SINGLE_TRIPWIRE) {
    processToFSingleTripwire();
  } else if (tofMode == TOF_DUAL_ZONE) {
    processToFDualZone();
  }
  
  delay(1);
}
