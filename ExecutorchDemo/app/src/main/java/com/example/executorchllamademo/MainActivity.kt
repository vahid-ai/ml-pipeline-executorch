package com.example.executorchllamademo

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.executorchllamademo.network.FeastApiClient
import com.example.executorchllamademo.network.FeatureResponse
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private lateinit var entityIdInput: EditText
    private lateinit var queryButton: Button
    private lateinit var progressBar: ProgressBar
    private lateinit var resultText: TextView
    private lateinit var feastClient: FeastApiClient

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize views
        entityIdInput = findViewById(R.id.entityIdInput)
        queryButton = findViewById(R.id.queryFeastButton)
        progressBar = findViewById(R.id.progressBar)
        resultText = findViewById(R.id.resultText)

        // Initialize API client
        feastClient = FeastApiClient()

        // Set button click listener
        queryButton.setOnClickListener {
            queryFeastFeatures()
        }

        // Optional: Test connection on startup
        testConnection()
    }

    private fun queryFeastFeatures() {
        val entityId = entityIdInput.text.toString().trim()

        if (entityId.isEmpty()) {
            Toast.makeText(this, "Please enter an Entity ID", Toast.LENGTH_SHORT).show()
            return
        }

        // Show loading state
        showLoading(true)

        // Define features you want to query
        val features = listOf(
            "feature_1",
            "feature_2",
            "feature_3"
        )

        // Optional: Add additional parameters
        val additionalParams = mapOf(
            "scale_features" to true,
            "scale_factor" to 1.5
        )

        lifecycleScope.launch {
            val result = feastClient.getFeatures(entityId, features, additionalParams)

            showLoading(false)

            result.fold(
                onSuccess = { response ->
                    displayResults(response)
                    Toast.makeText(this@MainActivity, "Features retrieved successfully!", Toast.LENGTH_SHORT).show()
                },
                onFailure = { exception ->
                    displayError(exception)
                    Toast.makeText(this@MainActivity, "Error: ${exception.message}", Toast.LENGTH_LONG).show()
                }
            )
        }
    }

    private fun testConnection() {
        lifecycleScope.launch {
            val result = feastClient.healthCheck()
            result.fold(
                onSuccess = { isHealthy ->
                    if (isHealthy) {
                        resultText.text = "✅ Connected to Feast server"
                    } else {
                        resultText.text = "❌ Feast server not responding"
                    }
                },
                onFailure = { exception ->
                    resultText.text = "❌ Connection failed: ${exception.message}"
                }
            )
        }
    }

    private fun displayResults(response: FeatureResponse) {
        val formattedResult = buildString {
            appendLine("🎯 FEAST FEATURES RETRIEVED")
            appendLine("=".repeat(40))
            appendLine()

            appendLine("📊 Features:")
            response.features.forEach { (key, value) ->
                appendLine("  $key: $value")
            }

            appendLine()
            appendLine("ℹ️ Metadata:")
            response.metadata.forEach { (key, value) ->
                appendLine("  $key: $value")
            }

            appendLine()
            appendLine("✅ Query completed at: ${System.currentTimeMillis()}")
        }

        resultText.text = formattedResult
    }

    private fun displayError(exception: Throwable) {
        val errorMessage = buildString {
            appendLine("❌ ERROR OCCURRED")
            appendLine("=".repeat(40))
            appendLine()
            appendLine("Error: ${exception.message}")
            appendLine()
            appendLine("Possible causes:")
            appendLine("• MacBook server not running")
            appendLine("• Wrong IP address in FeastApiClient")
            appendLine("• Network connectivity issues")
            appendLine("• Firewall blocking connection")
            appendLine()
            appendLine("Timestamp: ${System.currentTimeMillis()}")
        }

        resultText.text = errorMessage
    }

    private fun showLoading(show: Boolean) {
        if (show) {
            progressBar.visibility = ProgressBar.VISIBLE
            queryButton.isEnabled = false
            queryButton.text = "Querying..."
        } else {
            progressBar.visibility = ProgressBar.GONE
            queryButton.isEnabled = true
            queryButton.text = "Query Feast Features"
        }
    }
}