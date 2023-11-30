// Made by some great people...
// They are in SAE and TGW
// They are legends known by the names Ritwik, Sahil and Parth.
// You won't be able to find out who wrote this XD


#include <BasicLinearAlgebra.h>
#include <math.h>
using namespace BLA;


BLA::Matrix<2, 2> Q = {pow(10,-5),0,0,pow(10,-4)};
BLA::Matrix<1, 1> r = {0.004};
BLA::Matrix<2, 2> p = {0.025,0,0,0.01};
BLA::Matrix<2, 2> eye = {1,0,0,1};
const float Qn_rated    = 4.81 * 3600;
const float R_o = 1.611*pow(10,-4);
const float R = 0.116;
const float C = 10.386;
const float tau = R*C;
const float a1 = exp(-5/tau);
const float b1 = R*(1 - exp(-5/tau));
BLA::Matrix<2, 2> A = {01,0,0,a1};
BLA::Matrix<2, 1> B = {-1*0.9/(Qn_rated*3600),b1};

BLA::Matrix<2, 1> X = {1.0,0};
BLA::Matrix<2, 2> P = p;

const int relay1 = 12;
const int relay2 = 4;
const int current_data_pin = A0;
const int voltage_data_pin = A5;

float R1 = 997;
float R2 = 995;
float Rl = 0.9;
float OCV;
float dOCV;
float SOC;
float V_1;
float TerminalVoltage;
float error;

float voltage_factor = (R1+R2)/R1;
float ref_voltage = 5.1;
float ref_current = 5.00;

char current_code = '1';
char ocv_code = '2';

float voltage_sum = 0;
float current_sum = 0;

void setup() {
  // put your setup code here, to run once:
  pinMode(relay1, OUTPUT);
  pinMode(relay2, OUTPUT);
  // analogReference(EXTERNAL);
  Serial.begin(115200);
}


float measure_current(){
  digitalWrite(relay1, LOW);
  digitalWrite(relay2, LOW);
  float current_sum = 0;
  delay(1000);
  int count = 0;
  while (count<1000){
    float vol = (analogRead(current_data_pin)/1024.0000)*ref_current;
    float current = vol/Rl;
    current_sum += current;
    count++;
  }
  float avg_current = current_sum/1000;
  return avg_current;
}


float measure_ocv(){
  digitalWrite(relay1, HIGH);
  digitalWrite(relay2, HIGH);
  float voltage_sum = 0;
  delay(1000);
  int count = 0;
  while (count<1000){
    float raw_vol = analogRead(voltage_data_pin);
    float voltage = (raw_vol/1024.0000)*ref_voltage*voltage_factor;
    voltage_sum += voltage;
    count++;
  }
  digitalWrite(relay1, LOW);
  digitalWrite(relay2, LOW);
  float avg_voltage = voltage_sum/1000;
  return avg_voltage;
}
float polyval(float x){
 OCV = 9.82360408*pow(10,2)*pow(x,10) + (-4.94174943)*pow(10,3)*pow(x,9) + 1.04336810*pow(10,4)*pow(x,8) -1.19093328*pow(10,4)*pow(x,7)
      + 7.81994257*pow(10,3)*pow(x,6) + (-2.81698643)*pow(10,3)*pow(x,5) + 3.91394271*pow(10,2)*pow(x,4) + 71.9062848*pow(x,3)
       -37.2498049*pow(x,2) + 7.60091818*pow(x,1)  + 2.49369003;
 return OCV;
}
float polyder(float x){
  dOCV =  9.82360408*pow(10,3)*pow(x,9) - 4.44757449*pow(10,4)*pow(x,8) +  8.34694483*pow(10,4)*pow(x,7) - 8.33653293*pow(10,4)*pow(x,6) + 
        4.69196554*pow(10,4)*pow(x,5) + -1.40849322*pow(10,4)*pow(x,4) + 1.56557708*pow(10,3)*pow(x,3) +  2.15718854*pow(10,2)*pow(x,2) +
       -74.4996097*x +  7.60091818;
  return dOCV;
}
void EKF(float V,float I){
  SOC = X(0, 0);
  V_1  = X(1, 0);
  OCV = polyval(SOC);
  TerminalVoltage = OCV - (R_o*I) - V_1;
  dOCV = polyder(SOC);
  BLA::Matrix<1, 2> C = {dOCV,-1};
  error = V - TerminalVoltage;
  X = (A*X) + (B*I);
  P = (A*P*(~A)) + Q;
  BLA::Matrix<2, 1> K = (P*(~C))/(((C*P*(~C) + r))(0));
  X += (K*error);
  P = ((eye - K*C)*P);
}


void loop() {
  float Current = measure_current();
  float OCV = measure_ocv();
  EKF(OCV, Current);
  Serial.print(Current, 4);
  Serial.print("$");
  Serial.print(OCV, 4);
  Serial.print("$");
  Serial.println(X(0,0), 4);
  delay(3580);
}
