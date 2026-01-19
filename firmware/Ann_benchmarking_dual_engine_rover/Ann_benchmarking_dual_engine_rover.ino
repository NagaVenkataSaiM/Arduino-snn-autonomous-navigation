/*
 * FINAL FIRMWARE: FULL TELEMETRY
 * Sends: SNN_Time, ANN_Time, Red, Green, Blue, SNN_Action, ANN_Action
 */

#include <AFMotor.h>

#define S0 A0
#define S1 A1
#define S2 A2
#define S3 A3
#define SENSOR_OUT A4

AF_DCMotor motorRight1(1);  
AF_DCMotor motorLeft1(2);  
AF_DCMotor motorLeft(3);  
AF_DCMotor motorRight(4);

const int N_IN = 3;  
const int DURATION_STEPS = 12; 
volatile int anti_opt_sink = 0; 

// Weights (Standard)
float w_ctrl[3][3] = { {26.1, 1.9, 1.8}, {1.5, 26.2, 2.2}, {1.7, 1.5, 26.1} };
float w_rstdp[3][3] = { {1.0, 0.2, 0.7}, {0.4, 1.0, 0.3}, {0.2, 0.3, 1.0} };
float v_ctrl[3] = {0}; float v_rs[3] = {0};
float v_thr_ctrl = 25.0; float v_dec_ctrl = 0.8;
float v_thr_rs = 4.0; float v_dec_rs = 0.9; float inhib_rs = 5.0; float scale_rs = 3.5;

float w_hid[3][4] = { {2.5,-1.2,0.5,1.1}, {-0.8,3.1,-0.5,0.2}, {0.1,-0.9,2.8,1.5} };
float b_hid[4] = {-0.5, -0.1, 0.2, -0.3};
float w_out[4][3] = { {1.5,-0.5,0.2}, {-1.2,2.8,-0.1}, {0.5,-1.5,3.2}, {0.8,0.2,-0.5} };

int currentState = 0; int proposedState = -1; int consecutiveCount = 0; const int CONFIDENCE_NEEDED = 4;

void setup() {
  Serial.begin(115200);
  pinMode(S0, OUTPUT); pinMode(S1, OUTPUT);
  pinMode(S2, OUTPUT); pinMode(S3, OUTPUT);
  pinMode(SENSOR_OUT, INPUT);
  digitalWrite(S0, HIGH); digitalWrite(S1, LOW); 
  motorLeft.setSpeed(150); motorRight.setSpeed(150); motorLeft1.setSpeed(150); motorRight1.setSpeed(150);
  motorLeft.run(RELEASE); motorRight.run(RELEASE); motorLeft1.run(RELEASE); motorRight1.run(RELEASE);
}

float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

void get_sensor_input(float* input_array) {
  int r, g, b;
  digitalWrite(S2, LOW); digitalWrite(S3, LOW);
  r = pulseIn(SENSOR_OUT, LOW, 20000); if(r==0) r=3000;
  digitalWrite(S2, HIGH); digitalWrite(S3, HIGH);
  g = pulseIn(SENSOR_OUT, LOW, 20000); if(g==0) g=3000;
  digitalWrite(S2, LOW); digitalWrite(S3, HIGH);
  b = pulseIn(SENSOR_OUT, LOW, 20000); if(b==0) b=3000;

  float nr = constrain(map(r, 20, 600, 100, 0) / 100.0, 0.0, 1.0);
  float ng = constrain(map(g, 20, 600, 100, 0) / 100.0, 0.0, 1.0);
  float nb = constrain(map(b, 20, 600, 100, 0) / 100.0, 0.0, 1.0);

  float min_val = nr;
  if (ng < min_val) min_val = ng;
  if (nb < min_val) min_val = nb;
  input_array[0] = nr - min_val; input_array[1] = ng - min_val; input_array[2] = nb - min_val;
}

void apply_state() {
  if (currentState == 0) { motorLeft.run(RELEASE); motorRight.run(RELEASE); motorLeft1.run(RELEASE); motorRight1.run(RELEASE); }
  else if (currentState == 1) { motorLeft.run(FORWARD); motorRight.run(FORWARD); motorLeft1.run(FORWARD); motorRight1.run(FORWARD);}
  else if (currentState == 2) { motorLeft.run(BACKWARD); motorRight.run(BACKWARD); motorLeft1.run(BACKWARD); motorRight1.run(BACKWARD);}
}

void loop() {
  float inputs[N_IN];
  get_sensor_input(inputs);
  bool is_nocolor = (inputs[0] + inputs[1] + inputs[2]) < 0.15;

  // 1. SNN BENCHMARK
  unsigned long t_start_snn = micros();
  int snn_winner = -1;
  
  if (is_nocolor) {
    delayMicroseconds(5); 
  } else {
    int s_ctrl[3] = {0}; int s_rs[3] = {0};
    for(int i=0; i<3; i++) { v_ctrl[i]=0; v_rs[i]=0; }
    for (int t = 0; t < DURATION_STEPS; t++) {
      // Control
      float c_ctrl[3] = {0};
      for(int j=0; j<3; j++) {
        for(int i=0; i<3; i++) c_ctrl[j] += inputs[i] * w_ctrl[i][j];
        v_ctrl[j] = (v_ctrl[j] * v_dec_ctrl) + c_ctrl[j];
        if(v_ctrl[j] >= v_thr_ctrl) { s_ctrl[j]++; v_ctrl[j]=0; }
      }
      // Research
      float c_rs[3] = {0};
      int winner_t = -1; float max_v = -999;
      for(int j=0; j<3; j++) {
        for(int i=0; i<3; i++) c_rs[j] += (inputs[i] * scale_rs) * w_rstdp[i][j];
        v_rs[j] = (v_rs[j] * v_dec_rs) + c_rs[j];
        if(v_rs[j] > max_v) { max_v = v_rs[j]; winner_t = j; }
      }
      if(winner_t != -1 && max_v >= v_thr_rs) {
        s_rs[winner_t]++;
        for(int k=0; k<3; k++) v_rs[k] -= inhib_rs;
        v_rs[winner_t] = 0;
      }
    }
    int max_s = 0;
    for(int i=0; i<3; i++) { if(s_ctrl[i] > max_s) { max_s = s_ctrl[i]; snn_winner = i; } }
  }
  unsigned long duration_snn = micros() - t_start_snn;

  // 2. ANN BENCHMARK
  unsigned long t_start_ann = micros();
  float h_act[4];
  for(int j=0; j<4; j++) {
    float sum = 0;
    for(int i=0; i<3; i++) sum += inputs[i] * w_hid[i][j];
    h_act[j] = sigmoid(sum + b_hid[j]); 
  }
  float o_act[3];
  for(int j=0; j<3; j++) {
    float sum = 0;
    for(int i=0; i<4; i++) sum += h_act[i] * w_out[i][j];
    o_act[j] = sigmoid(sum); 
  }
  int ann_winner = -1; float max_prob = -1.0;
  for(int i=0; i<3; i++) {
      if(o_act[i] > max_prob) { max_prob = o_act[i]; ann_winner = i; }
  }
  anti_opt_sink = ann_winner;
  unsigned long duration_ann = micros() - t_start_ann;

  // 3. MOTOR CONTROL
  if (snn_winner != -1) {
    if (snn_winner == proposedState) consecutiveCount++;
    else { proposedState = snn_winner; consecutiveCount = 1; }
    if (consecutiveCount >= CONFIDENCE_NEEDED) { currentState = proposedState; consecutiveCount = 0; }
  }
  apply_state();

  // 4. DATA EXPORT (7 Values)
  // Format: SNN_T, ANN_T, R, G, B, SNN_ACT, ANN_ACT
  Serial.print(duration_snn); Serial.print(",");
  Serial.print(duration_ann); Serial.print(",");
  Serial.print(inputs[0]); Serial.print(",");
  Serial.print(inputs[1]); Serial.print(",");
  Serial.print(inputs[2]); Serial.print(",");
  Serial.print(currentState); Serial.print(","); // SNN Action (Physical)
  Serial.println(ann_winner); // ANN Action (Theoretical)
  
  delay(50);
}