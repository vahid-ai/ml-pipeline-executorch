package com.example.executorchllamademo.network

import com.google.gson.Gson
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.util.concurrent.TimeUnit

data class FeatureRequest(
    val entity_id: String,
    val features: List<String>,
    val additional_params: Map<String, Any> = emptyMap()
)

data class FeatureResponse(
    val features: Map<String, Any>,
    val metadata: Map<String, Any>
)

class FeastApiClient {
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build()

    private val gson = Gson()
    private val baseUrl = "http://192.168.x.x:8000" // Replace with your MacBook IP

    suspend fun getFeatures(
        entityId: String,
        features: List<String>,
        additionalParams: Map<String, Any> = emptyMap()
    ): Result<FeatureResponse> = withContext(Dispatchers.IO) {
        try {
            val request = FeatureRequest(entityId, features, additionalParams)
            val requestBody = gson.toJson(request)
                .toRequestBody("application/json".toMediaType())

            val httpRequest = Request.Builder()
                .url("$baseUrl/get-features")
                .post(requestBody)
                .build()

            val response = client.newCall(httpRequest).execute()

            if (response.isSuccessful) {
                val responseBody = response.body?.string()
                val featureResponse = gson.fromJson(responseBody, FeatureResponse::class.java)
                Result.success(featureResponse)
            } else {
                Result.failure(Exception("Server error: ${response.code} - ${response.message}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    // Add the missing healthCheck method
    suspend fun healthCheck(): Result<Boolean> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$baseUrl/health")  // Adjust endpoint as needed
                .get()
                .build()

            val response = client.newCall(request).execute()
            Result.success(response.isSuccessful)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}