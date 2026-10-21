#include <Arduino.h>
#include <WiFi.h>
#include <ArduinoOTA.h>
#include "esp_camera.h"
#include "esp_http_server.h" // lets esp32 act like a mini website (so much bloat omg)
#include "../include/secrets.h"
#include <ESP32Servo.h>

Servo servo;
const int SERVO_PIN = 2;

// motor driver
#define LEFT_FWD 12
#define LEFT_BWD 13
#define RIGHT_FWD 15
#define RIGHT_BWD 14

// esp32-cam pinout for server
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

httpd_handle_t stream_httpd = NULL;

// fires on GET request from the browser
// req is ptr to request
static esp_err_t feed_handler(httpd_req_t* req) {
  esp_err_t status = OK;
  char header_buffer[64]; // hold text header from each frame

  // tells the browser to keep connection open and listen for new images (boundary is what separates img1 from img2)
  status = httpd_resp_set_type(req, "multipart/x-mixed-replace;boundary=YOUNGMULTI676767");
  if (status != ESP_OK) {return status;}

  // infinite listening loop
  while(true) {
    // keep OTA listening so we dont lock ourselves out
    ArduinoOTA.handle(); 

    // snap a pic and get it in RAM 
    camera_fb_t *frame_buffer = esp_camera_fb_get();
    if (!frame_buffer) {continue;} // if snap failed, restart the loop

    // build the header telling the browser the file size
    size_t header_len = snprintf(header_buffer, 64, "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", frame_buffer->len);

    // send header -> send image bytes -> boundary
    status = httpd_resp_send_chunk(req, (const char*)header_buffer, header_len);
    if(status == ESP_OK) {
      status = httpd_resp_send_chunk(req, (const char*)frame_buffer->buf, frame_buffer->len);
    }
    if(status == ESP_OK) {
      status = httpd_resp_send_chunk(req, "\r\n--YOUNGMULTI676767\r\n", 22);
    }

    // delete the pic from esp32-cam RAM
    esp_camera_fb_return(frame_buffer);
    
    // if sending failed then close the loop.
    if (status != ESP_OK) {break;}
  }
  
  return status;
}

// takes a request and sets motors with analog_write
static esp_err_t steering_handler(httpd_req_t* req) {
  char buffer[64];

  // save request into buffer
  if (httpd_req_get_url_query_str(req, buffer, sizeof(buffer)) == ESP_OK) {
    char value[10]; 
    
    // find dir= in query and save parameter into value
    if (httpd_query_key_value(buffer, "dir", value, sizeof(value)) == ESP_OK) {
      int speed = 76; // big PWM will beat the shit out of low quality TT motors

      if (strcmp(value, "forward") == 0) {
        analogWrite(LEFT_FWD, speed);
        analogWrite(RIGHT_FWD, speed);
        
        analogWrite(LEFT_BWD, 0);
        analogWrite(RIGHT_BWD, 0);
      }
      
      else if (strcmp(value, "backward") == 0) {
        analogWrite(LEFT_BWD, speed);
        analogWrite(RIGHT_BWD, speed);

        analogWrite(LEFT_FWD, 0);
        analogWrite(RIGHT_FWD, 0);
      }
      
      else if (strcmp(value, "left") == 0) {
        analogWrite(LEFT_BWD, speed);
        analogWrite(RIGHT_FWD, speed);

        analogWrite(LEFT_FWD, 0);
        analogWrite(RIGHT_BWD, 0);
      }
      
      else if (strcmp(value, "right") == 0) {
        analogWrite(LEFT_FWD, speed);
        analogWrite(RIGHT_BWD, speed);

        analogWrite(LEFT_BWD, 0);
        analogWrite(RIGHT_FWD, 0);
      }
      
      else if (strcmp(value, "stop") == 0) {
        analogWrite(LEFT_FWD, 0);
        analogWrite(LEFT_BWD, 0);
        analogWrite(RIGHT_FWD, 0);
        analogWrite(RIGHT_BWD, 0);
      }
    }
  }

  httpd_resp_sendstr(req, "OK");

  return ESP_OK;
}

void start_camera_server() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG(); // load up config for web server
  config.server_port = 80;
  httpd_uri_t feed_uri = { .uri = "/", .method = HTTP_GET, .handler = feed_handler, .user_ctx = NULL }; // if user uses http_get, run stream_handler function
  // so if u put in the IP of esp32 the browser sends GET "/" and starts firing frames in a while (true) loop
  httpd_uri_t steer_uri  = { .uri = "/control", .method = HTTP_GET, .handler = steering_handler, .user_ctx = NULL }; // if user uses the control/ flag.
  
  if (httpd_start(&stream_httpd, &config) == ESP_OK) { // boot up the server
    httpd_register_uri_handler(stream_httpd, &feed_uri); // attach the feed rule
    httpd_register_uri_handler(stream_httpd, &steer_uri); // attach the steering rule
  }
}

void setup() {
  Serial.begin(115200);

  // init pins and set to low
  pinMode(LEFT_FWD, OUTPUT); 
  pinMode(LEFT_BWD, OUTPUT);
  pinMode(RIGHT_FWD, OUTPUT); 
  pinMode(RIGHT_BWD, OUTPUT);
  analogWrite(LEFT_FWD, 0); 
  analogWrite(LEFT_BWD, 0);
  analogWrite(RIGHT_FWD, 0); 
  analogWrite(RIGHT_BWD, 0);

  ESP32PWM::allocateTimer(3);
  servo.setPeriodHertz(50);
  servo.attach(SERVO_PIN, 500, 2400);
  servo.write(0);

  // connect to wifi
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_NAME, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }

  IPAddress ip = WiFi.localIP();
  int last_piece = ip[3];
  
  Serial.print("Connected. IP ends with: ");
  Serial.println(last_piece);

  for(int i = 0; i < last_piece; i++) {
    servo.write(45);
    delay(150);
    servo.write(0); 
    delay(150);
  }

  // initialize OTA
  ArduinoOTA.begin();

  // initialize camera
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_VGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  esp_camera_init(&config);
  
  // launch web server and listen
  start_camera_server();
}

void loop() {
  // if nobody is listening then loop() is running, so listen to OTA.
  ArduinoOTA.handle();
}