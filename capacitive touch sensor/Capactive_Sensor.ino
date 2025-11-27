
#define Frequency 2500
#define trig 3
#define out 2
#define in A5
unsigned int period, quarPeriod, halfPeriod; //us
unsigned long lastTime,cutOff;
int phase, read, count, touched;

void setup() {
  cutOff = 900;
  period = 1000000/Frequency; //us
  halfPeriod = period/2; //us
  quarPeriod = halfPeriod/2; //us
  lastTime = micros();
  phase = 0;
  read = 0;
  count = 0;
  touched = 0;
  pinMode(trig,OUTPUT);
  pinMode(out,OUTPUT);
  pinMode(in,INPUT);
  digitalWrite(trig,LOW);
  Serial.begin(9600);
}

void loop() {
  if (phase == 0 && micros()-lastTime >= halfPeriod){
    lastTime = micros();
    digitalWrite(trig,HIGH);
    phase +=1;
  }
  if(phase == 1 && micros()-lastTime >= quarPeriod){
    read = analogRead(in);
    
    read = (read > cutOff);
    if (read == 1){
      touched +=1;
      if (touched <10) read = 0;
    }
    else touched = 0;
    if (count%1000 == 0) Serial.println(read);
    digitalWrite(out,(read)?HIGH:LOW);
    
    phase +=1;
    count +=1;
  }
  if (phase == 2 && micros()-lastTime >= halfPeriod){
    lastTime = micros();
    digitalWrite(trig,LOW);
    phase = 0;
  }

}
