# 📱 AI Interview Coach - Android App & APK Guide

This directory contains the complete native **Android Studio** project for the **AI-Powered Interview Coach** mobile application.

It is pre-configured with:
- ✅ **WebRTC Camera & Microphone permissions** for posture detection and speech evaluation.
- ✅ **Dynamic runtime permission requests** (Android 6.0+).
- ✅ **File upload handling** for resumes and audio recordings.
- ✅ **Pull-to-refresh** and offline fallback screen with a retry button.
- ✅ **Network Security Config** supporting both HTTPS cloud endpoints and HTTP local development.

---

## 🚀 Quick Start: Building the APK

### Step 1: Configure Your Backend Server URL
Open `app/src/main/res/values/strings.xml`:
```xml
<string name="default_web_url">https://your-cloud-app.onrender.com</string>
```
- **For Cloud Hosting (Render, Railway, Hugging Face, VPS):**
  Enter your live HTTPS URL (e.g. `https://my-interview-coach.onrender.com`).
- **For Local Testing on Android Emulator:**
  Use `http://10.0.2.2:5000` (this special IP connects the Android emulator to your PC's `localhost:5000`).
- **For Testing on a Physical Phone via Wi-Fi:**
  Find your computer's local IP address (run `ipconfig` in terminal, e.g. `192.168.1.15`) and set:
  `http://192.168.1.15:5000` (make sure Flask is run with `python app.py` listening on `0.0.0.0`).

---

### Step 2: Open in Android Studio
1. Launch **Android Studio**.
2. Click **File** > **Open...**
3. Navigate to and select the `android` folder in this repository:
   ```
   AI-Powered-Interview-Coach/android
   ```
4. Android Studio will open the project and automatically download the required Gradle build tools and dependencies.

---

### Step 3: Build the APK File
1. In the Android Studio menu, click:
   **Build** > **Build Bundle(s) / APK(s)** > **Build APK(s)**
2. Once Gradle finishes building (typically 1–2 minutes), a notification popup will appear at the bottom right:
   `APK(s) generated successfully for 1 module.`
3. Click the **locate** link in the popup, or find your APK directly at:
   ```
   android/app/build/outputs/apk/debug/app-debug.apk
   ```

---

### Step 4: Install the APK on Your Android Device
You can install the APK onto your Android phone using any of these methods:
- **Direct USB Transfer / Share:** Send `app-debug.apk` to your phone via WhatsApp, Google Drive, or USB cable, tap on the file, and select **Install** (allow "Install from unknown sources" if prompted).
- **Via ADB:** Connect phone via USB with USB Debugging enabled, and run:
  ```powershell
  adb install app/build/outputs/apk/debug/app-debug.apk
  ```
- **Run Directly in Android Studio:** Connect your phone or start an Android Emulator and click the green **Run (▶)** button in Android Studio.

---

## ⚙️ Project Structure
```
android/
├── build.gradle                              # Top-level build configuration
├── settings.gradle                           # Project settings
├── gradle.properties                         # AndroidX & JVM parameters
├── gradle/wrapper/gradle-wrapper.properties  # Gradle 8.2 wrapper config
└── app/
    ├── build.gradle                          # App module (targetSdk 34, minSdk 24)
    ├── proguard-rules.pro                    # Proguard rules
    └── src/main/
        ├── AndroidManifest.xml               # Permissions (CAMERA, RECORD_AUDIO, INTERNET)
        ├── java/com/aicoach/interviewcoach/
        │   └── MainActivity.java             # Fullscreen WebView + WebRTC permissions
        └── res/
            ├── drawable/ic_launcher.xml      # App launcher icon
            ├── layout/activity_main.xml      # WebView + SwipeRefresh + Error UI
            ├── values/
            │   ├── colors.xml                # Theme color palette
            │   ├── strings.xml               # App title & default_web_url
            │   └── themes.xml                # Fullscreen theme
            └── xml/
                └── network_security_config.xml # Cleartext traffic permissions
```
