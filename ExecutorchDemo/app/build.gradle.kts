/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

plugins {
  id("com.android.application")
  id("org.jetbrains.kotlin.android")
}

android {
  namespace = "com.example.executorchllamademo"
  compileSdk = 34

  defaultConfig {
    applicationId = "com.example.executorchllamademo"
    minSdk = 28
    targetSdk = 34
    versionCode = 1
    versionName = "1.0"

    testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    vectorDrawables { useSupportLibrary = true }
    externalNativeBuild { cmake { cppFlags += "" } }
  }

  buildTypes {
    release {
      isMinifyEnabled = false
      proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
    }
  }
  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_1_8
    targetCompatibility = JavaVersion.VERSION_1_8
  }
  kotlinOptions { jvmTarget = "1.8" }
  buildFeatures { compose = true }
  composeOptions { kotlinCompilerExtensionVersion = "1.5.1" }
  packaging { resources { excludes += "/META-INF/{AL2.0,LGPL2.1}" } }
  packagingOptions {
    jniLibs {
      useLegacyPackaging = true
    }
  }
}

dependencies {
  implementation ("com.squareup.okhttp3:okhttp:4.11.0")
  implementation ("com.google.code.gson:gson:2.10.1")
  implementation ("androidx.lifecycle:lifecycle-runtime-ktx:2.8.6")
  implementation ("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
  implementation("androidx.core:core-ktx:1.13.1")
  implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.6")
  implementation("androidx.activity:activity-compose:1.9.2")
  implementation(platform("androidx.compose:compose-bom:2024.06.00"))
  implementation("androidx.compose.ui:ui")
  implementation("androidx.compose.ui:ui-graphics")
  implementation("androidx.compose.ui:ui-tooling-preview")
  implementation("androidx.compose.material3:material3")
  implementation("androidx.appcompat:appcompat:1.7.0")
  implementation("androidx.camera:camera-core:1.4.2")
  implementation("androidx.constraintlayout:constraintlayout:2.2.1")
  implementation("com.facebook.fbjni:fbjni:0.5.1")
  implementation("com.google.code.gson:gson:2.10.1")
  implementation("org.pytorch:executorch-android:0.6.0-rc1")
  implementation("com.google.android.material:material:1.12.0")
  implementation("androidx.activity:activity:1.9.2")
  implementation("org.json:json:20250107")
  testImplementation("junit:junit:4.13.2")
  androidTestImplementation("androidx.test.ext:junit:1.2.1")
  androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")
  androidTestImplementation(platform("androidx.compose:compose-bom:2024.06.00"))
  androidTestImplementation("androidx.compose.ui:ui-test-junit4")
  debugImplementation("androidx.compose.ui:ui-tooling")
  debugImplementation("androidx.compose.ui:ui-test-manifest")
}

tasks.register("setup") {
  doFirst {
    exec {
      commandLine("sh", "examples/demo-apps/android/LlamaDemo/setup.sh")
      workingDir("../../../../../")
    }
  }
}

tasks.register("setupQnn") {
  doFirst {
    exec {
      commandLine("sh", "examples/demo-apps/android/LlamaDemo/setup-with-qnn.sh")
      workingDir("../../../../../")
    }
  }
}

tasks.register("download_prebuilt_lib") {
  doFirst {
    exec {
      commandLine("sh", "examples/demo-apps/android/LlamaDemo/download_prebuilt_lib.sh")
      workingDir("../../../../../")
    }
  }
}
