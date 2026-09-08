package com.aicoach.interviewcoach;

import android.Manifest;
import android.annotation.SuppressLint;
import android.annotation.TargetApi;
import android.app.Activity;
import android.app.DownloadManager;
import android.content.ActivityNotFoundException;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.ViewGroup;
import android.os.Environment;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.view.View;
import android.webkit.CookieManager;
import android.webkit.DownloadListener;
import android.webkit.JavascriptInterface;
import android.webkit.PermissionRequest;
import android.webkit.URLUtil;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.Toast;

import androidx.activity.OnBackPressedCallback;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {

    private static final int PERMISSION_REQUEST_CODE = 1001;
    private static final int FILE_CHOOSER_REQUEST_CODE = 1002;
    private static final int SPEECH_PERMISSION_REQUEST_CODE = 1003;

    private WebView webView;
    private SwipeRefreshLayout swipeRefreshLayout;
    private ProgressBar progressBar;
    private LinearLayout errorLayout;
    private Button retryButton;
    private Button changeUrlButton;
    private LinearLayout splashLayout;

    private PermissionRequest pendingWebPermissionRequest;
    private ValueCallback<Uri[]> fileUploadCallback;
    private String targetUrl;

    // Native Speech Recognition
    private SpeechRecognizer speechRecognizer;
    private boolean isListening = false;
    private final Handler speechHandler = new Handler(Looper.getMainLooper());
    private StringBuilder accumulatedTranscript = new StringBuilder();
    private String lastCommittedUtterance = "";
    private String currentTargetElementId;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        SharedPreferences prefs = getSharedPreferences("app_settings", MODE_PRIVATE);
        targetUrl = prefs.getString("server_url", getString(R.string.default_web_url));

        initViews();
        setupWebView();
        setupSwipeRefresh();
        setupBackNavigation();
        checkAndRequestPermissions();

        loadUrl(targetUrl);
    }

    private void initViews() {
        webView = findViewById(R.id.webView);
        swipeRefreshLayout = findViewById(R.id.swipeRefreshLayout);
        progressBar = findViewById(R.id.progressBar);
        errorLayout = findViewById(R.id.errorLayout);
        retryButton = findViewById(R.id.retryButton);
        changeUrlButton = findViewById(R.id.changeUrlButton);
        splashLayout = findViewById(R.id.splashLayout);

        retryButton.setOnClickListener(v -> {
            errorLayout.setVisibility(View.GONE);
            if (splashLayout != null) {
                splashLayout.setAlpha(1f);
                splashLayout.setVisibility(View.VISIBLE);
            }
            webView.setVisibility(View.VISIBLE);
            loadUrl(targetUrl);
        });

        if (changeUrlButton != null) {
            changeUrlButton.setOnClickListener(v -> showChangeUrlDialog());
        }
    }

    private void showChangeUrlDialog() {
        android.app.AlertDialog.Builder builder = new android.app.AlertDialog.Builder(this);
        builder.setTitle("Configure Server URL");
        builder.setMessage("Enter the URL of your AI Interview Coach server (e.g. https://shauriya24-ai-powered-interview-coach.hf.space):");

        final android.widget.EditText input = new android.widget.EditText(this);
        input.setInputType(android.text.InputType.TYPE_CLASS_TEXT | android.text.InputType.TYPE_TEXT_VARIATION_URI);
        input.setText(targetUrl);
        input.setSelection(input.getText().length());

        android.widget.FrameLayout container = new android.widget.FrameLayout(this);
        android.widget.FrameLayout.LayoutParams params = new android.widget.FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        params.leftMargin = 50;
        params.rightMargin = 50;
        input.setLayoutParams(params);
        container.addView(input);
        builder.setView(container);

        builder.setPositiveButton("Save & Connect", (dialog, which) -> {
            String newUrl = input.getText().toString().trim();
            if (!newUrl.isEmpty()) {
                if (!newUrl.startsWith("http://") && !newUrl.startsWith("https://")) {
                    newUrl = "http://" + newUrl;
                }
                targetUrl = newUrl;
                SharedPreferences prefs = getSharedPreferences("app_settings", MODE_PRIVATE);
                prefs.edit().putString("server_url", targetUrl).apply();

                errorLayout.setVisibility(View.GONE);
                if (splashLayout != null) {
                    splashLayout.setAlpha(1f);
                    splashLayout.setVisibility(View.VISIBLE);
                }
                webView.setVisibility(View.VISIBLE);
                loadUrl(targetUrl);
            }
        });

        builder.setNegativeButton("Cancel", (dialog, which) -> dialog.cancel());
        builder.show();
    }

    @SuppressLint("SetJavaScriptEnabled")
    private void setupWebView() {
        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setDatabaseEnabled(true);
        settings.setAllowFileAccess(true);
        settings.setAllowContentAccess(true);
        settings.setMediaPlaybackRequiresUserGesture(false);
        settings.setUseWideViewPort(true);
        settings.setLoadWithOverviewMode(false);
        settings.setTextZoom(100);
        settings.setSupportZoom(true);
        settings.setBuiltInZoomControls(false);
        settings.setDisplayZoomControls(false);
        settings.setCacheMode(WebSettings.LOAD_DEFAULT);
        settings.setSupportMultipleWindows(true);
        settings.setJavaScriptCanOpenWindowsAutomatically(true);

        // User-Agent: mark as Android app for server-side detection if needed
        String defaultUA = settings.getUserAgentString();
        settings.setUserAgentString(defaultUA + " AIInterviewCoach-Android/1.0");

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            settings.setMixedContentMode(WebSettings.MIXED_CONTENT_ALWAYS_ALLOW);
        }

        // Register JS-to-Native bridge for speech recognition
        webView.addJavascriptInterface(new NativeSpeechBridge(), "AndroidSpeech");

        webView.setWebViewClient(new CustomWebViewClient());
        webView.setWebChromeClient(new CustomWebChromeClient());

        // Handle file downloads (PDF reports, mock interview videos, etc.)
        webView.setDownloadListener(new DownloadListener() {
            @Override
            public void onDownloadStart(String url, String userAgent, String contentDisposition, String mimetype, long contentLength) {
                downloadFile(url, userAgent, contentDisposition, mimetype);
            }
        });
    }

    private void downloadFile(String url, String userAgent, String contentDisposition, String mimetype) {
        try {
            DownloadManager.Request request = new DownloadManager.Request(Uri.parse(url));
            request.setMimeType(mimetype);

            // Forward cookies from WebView to DownloadManager for authenticated downloads
            String cookies = CookieManager.getInstance().getCookie(url);
            if (cookies != null) {
                request.addRequestHeader("Cookie", cookies);
            }
            request.addRequestHeader("User-Agent", userAgent);

            String filename = URLUtil.guessFileName(url, contentDisposition, mimetype);
            request.setTitle(filename);
            request.setDescription("Downloading file...");
            request.allowScanningByMediaScanner();
            request.setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED);
            request.setDestinationInExternalPublicDir(Environment.DIRECTORY_DOWNLOADS, filename);

            DownloadManager dm = (DownloadManager) getSystemService(Context.DOWNLOAD_SERVICE);
            if (dm != null) {
                dm.enqueue(request);
                Toast.makeText(MainActivity.this, "Downloading " + filename, Toast.LENGTH_SHORT).show();
            }
        } catch (Exception e) {
            // Fallback: open in external browser
            try {
                Intent browserIntent = new Intent(Intent.ACTION_VIEW, Uri.parse(url));
                startActivity(browserIntent);
            } catch (Exception ex) {
                Toast.makeText(MainActivity.this, "Unable to download file", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private void setupSwipeRefresh() {
        swipeRefreshLayout.setColorSchemeResources(R.color.primary, R.color.accent);
        swipeRefreshLayout.setOnRefreshListener(() -> {
            errorLayout.setVisibility(View.GONE);
            webView.setVisibility(View.VISIBLE);
            webView.reload();
        });
    }

    private void setupBackNavigation() {
        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) {
            @Override
            public void handleOnBackPressed() {
                if (webView.canGoBack()) {
                    webView.goBack();
                } else {
                    setEnabled(false);
                    getOnBackPressedDispatcher().onBackPressed();
                }
            }
        });
    }

    private void loadUrl(String url) {
        if (url != null && !url.trim().isEmpty()) {
            webView.loadUrl(url);
        }
    }

    private boolean checkAndRequestPermissions() {
        List<String> neededPermissions = new ArrayList<>();
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            neededPermissions.add(Manifest.permission.CAMERA);
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            neededPermissions.add(Manifest.permission.RECORD_AUDIO);
        }

        if (!neededPermissions.isEmpty()) {
            ActivityCompat.requestPermissions(this, neededPermissions.toArray(new String[0]), PERMISSION_REQUEST_CODE);
            return false;
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_REQUEST_CODE) {
            boolean allGranted = true;
            for (int result : grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    allGranted = false;
                    break;
                }
            }

            if (allGranted && pendingWebPermissionRequest != null) {
                pendingWebPermissionRequest.grant(pendingWebPermissionRequest.getResources());
                pendingWebPermissionRequest = null;
            } else if (!allGranted) {
                Toast.makeText(this, R.string.permission_camera_mic, Toast.LENGTH_LONG).show();
            }
        } else if (requestCode == SPEECH_PERMISSION_REQUEST_CODE) {
            boolean granted = grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED;
            if (granted) {
                // Retry speech start after permission grant
                webView.evaluateJavascript(
                    "if(window._pendingSpeechTarget) { AndroidSpeech.startListening(window._pendingSpeechTarget); }",
                    null
                );
            }
        }
    }

    // =========================================================================
    // Native Speech Recognition Bridge (JS <-> Android)
    // =========================================================================
    private class NativeSpeechBridge {

        @JavascriptInterface
        public boolean isAvailable() {
            return SpeechRecognizer.isRecognitionAvailable(MainActivity.this);
        }

        @JavascriptInterface
        public void startListening(final String targetElementId) {
            runOnUiThread(() -> {
                if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.RECORD_AUDIO)
                        != PackageManager.PERMISSION_GRANTED) {
                    webView.evaluateJavascript(
                        "window._pendingSpeechTarget = '" + targetElementId.replace("'", "\\'") + "';",
                        null
                    );
                    ActivityCompat.requestPermissions(MainActivity.this,
                        new String[]{Manifest.permission.RECORD_AUDIO},
                        SPEECH_PERMISSION_REQUEST_CODE);
                    return;
                }

                stopListeningInternal();

                accumulatedTranscript = new StringBuilder();
                lastCommittedUtterance = "";
                currentTargetElementId = targetElementId;
                speechRecognizer = SpeechRecognizer.createSpeechRecognizer(MainActivity.this);
                isListening = true;

                final Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
                intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
                intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.US.toString());
                intent.putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true);
                intent.putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1);

                speechRecognizer.setRecognitionListener(new RecognitionListener() {
                    @Override public void onReadyForSpeech(Bundle params) {
                        callJs("_nativeSpeechOnStart", targetElementId, "");
                    }

                    @Override public void onBeginningOfSpeech() {}
                    @Override public void onRmsChanged(float rmsdB) {}
                    @Override public void onBufferReceived(byte[] buffer) {}
                    @Override public void onEndOfSpeech() {}

                    @Override
                    public void onError(int error) {
                        // Error 7 = no match (silence), Error 6 = speech timeout -> seamless auto-restart
                        if ((error == SpeechRecognizer.ERROR_NO_MATCH || error == SpeechRecognizer.ERROR_SPEECH_TIMEOUT) && isListening) {
                            scheduleRestart(intent, targetElementId);
                            return;
                        }
                        isListening = false;
                        callJs("_nativeSpeechOnEnd", targetElementId, "");
                    }

                    @Override
                    public void onResults(Bundle results) {
                        if (results != null) {
                            List<String> matches = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
                            if (matches != null && !matches.isEmpty()) {
                                String utterance = matches.get(0).trim();
                                if (!utterance.isEmpty()) {
                                    // DEDUPLICATION: Prevent duplicate repeated phrases
                                    if (!utterance.equalsIgnoreCase(lastCommittedUtterance)) {
                                        String base = accumulatedTranscript.toString().trim();
                                        if (base.isEmpty() || !base.toLowerCase().endsWith(utterance.toLowerCase())) {
                                            if (accumulatedTranscript.length() > 0) {
                                                accumulatedTranscript.append(" ");
                                            }
                                            accumulatedTranscript.append(utterance);
                                            lastCommittedUtterance = utterance;
                                            callJs("_nativeSpeechOnResult", targetElementId, accumulatedTranscript.toString().trim());
                                        }
                                    }
                                }
                            }
                        }

                        // Auto-restart with safe delay so user can continue speaking naturally
                        if (isListening) {
                            scheduleRestart(intent, targetElementId);
                        }
                    }

                    @Override
                    public void onPartialResults(Bundle partialResults) {
                        if (partialResults != null) {
                            List<String> partial = partialResults.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
                            if (partial != null && !partial.isEmpty()) {
                                String part = partial.get(0).trim();
                                if (!part.isEmpty()) {
                                    String base = accumulatedTranscript.toString().trim();
                                    // Show real-time preview without duplicating or committing
                                    String preview = base.isEmpty() ? part : base + " " + part;
                                    callJs("_nativeSpeechOnPartial", targetElementId, preview);
                                }
                            }
                        }
                    }

                    @Override public void onEvent(int eventType, Bundle params) {}
                });

                speechRecognizer.startListening(intent);
            });
        }

        @JavascriptInterface
        public void stopListening() {
            runOnUiThread(() -> stopListeningInternal());
        }
    }

    private void scheduleRestart(Intent intent, String targetElementId) {
        if (!isListening || speechRecognizer == null) return;
        speechHandler.removeCallbacksAndMessages(null);
        speechHandler.postDelayed(() -> {
            if (isListening && speechRecognizer != null) {
                try {
                    speechRecognizer.cancel();
                    speechRecognizer.startListening(intent);
                } catch (Exception e) {
                    isListening = false;
                    callJs("_nativeSpeechOnEnd", targetElementId, "");
                }
            }
        }, 350);
    }

    private void stopListeningInternal() {
        isListening = false;
        speechHandler.removeCallbacksAndMessages(null);
        if (speechRecognizer != null) {
            try {
                speechRecognizer.stopListening();
                speechRecognizer.cancel();
                speechRecognizer.destroy();
            } catch (Exception ignored) {}
            speechRecognizer = null;
        }
        if (currentTargetElementId != null) {
            callJs("_nativeSpeechOnEnd", currentTargetElementId, "");
        }
    }

    private void callJs(String functionName, String targetId, String text) {
        String escapedText = text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n").replace("\r", "");
        String js = "javascript:if(typeof " + functionName + " === 'function'){" + functionName + "('" +
                targetId.replace("'", "\\'") + "','" + escapedText + "');}";
        runOnUiThread(() -> webView.loadUrl(js));
    }

    // =========================================================================
    // WebChromeClient
    // =========================================================================
    private class CustomWebChromeClient extends WebChromeClient {

        @Override
        public void onProgressChanged(WebView view, int newProgress) {
            if (newProgress < 100) {
                progressBar.setVisibility(View.VISIBLE);
                progressBar.setProgress(newProgress);
            } else {
                progressBar.setVisibility(View.GONE);
                swipeRefreshLayout.setRefreshing(false);
            }
        }

        @Override
        @TargetApi(Build.VERSION_CODES.LOLLIPOP)
        public void onPermissionRequest(final PermissionRequest request) {
            MainActivity.this.runOnUiThread(() -> {
                boolean hasCamera = ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
                boolean hasMic = ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED;

                if (hasCamera && hasMic) {
                    request.grant(request.getResources());
                } else {
                    pendingWebPermissionRequest = request;
                    checkAndRequestPermissions();
                }
            });
        }

        @Override
        public boolean onShowFileChooser(WebView webView, ValueCallback<Uri[]> filePathCallback, FileChooserParams fileChooserParams) {
            if (fileUploadCallback != null) {
                fileUploadCallback.onReceiveValue(null);
            }
            fileUploadCallback = filePathCallback;

            Intent intent = fileChooserParams.createIntent();
            try {
                startActivityForResult(intent, FILE_CHOOSER_REQUEST_CODE);
            } catch (Exception e) {
                fileUploadCallback = null;
                return false;
            }
            return true;
        }

        // Handle window.open() — open the URL in the same WebView or external browser
        @Override
        public boolean onCreateWindow(WebView view, boolean isDialog, boolean isUserGesture, android.os.Message resultMsg) {
            // Get the target URL from the hit test
            WebView.HitTestResult result = view.getHitTestResult();
            String url = result.getExtra();

            if (url != null) {
                // Check if it's a download/API URL (like PDF reports)
                if (url.contains("/api/reports/pdf") || url.contains("/api/mock-interview/")) {
                    // Trigger download via the system browser/download manager
                    try {
                        Intent browserIntent = new Intent(Intent.ACTION_VIEW, Uri.parse(url));
                        startActivity(browserIntent);
                    } catch (ActivityNotFoundException e) {
                        view.loadUrl(url);
                    }
                } else {
                    // Load in same WebView
                    view.loadUrl(url);
                }
            }
            return false;
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == FILE_CHOOSER_REQUEST_CODE) {
            if (fileUploadCallback == null) return;

            Uri[] results = null;
            if (resultCode == Activity.RESULT_OK && data != null) {
                String dataString = data.getDataString();
                if (dataString != null) {
                    results = new Uri[]{Uri.parse(dataString)};
                } else if (data.getClipData() != null) {
                    int count = data.getClipData().getItemCount();
                    results = new Uri[count];
                    for (int i = 0; i < count; i++) {
                        results[i] = data.getClipData().getItemAt(i).getUri();
                    }
                }
            }
            fileUploadCallback.onReceiveValue(results);
            fileUploadCallback = null;
        }
    }

    // =========================================================================
    // WebViewClient
    // =========================================================================
    private class CustomWebViewClient extends WebViewClient {

        @Override
        public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
            Uri uri = request.getUrl();
            String scheme = uri.getScheme();

            // Allow normal web traffic inside WebView
            if ("http".equalsIgnoreCase(scheme) || "https".equalsIgnoreCase(scheme)) {
                String path = uri.getPath();
                // Intercept PDF/video download URLs to open in external browser/downloader
                if (path != null && (path.contains("/api/reports/pdf") || path.contains("/api/mock-interview/"))) {
                    downloadFile(uri.toString(),
                        webView.getSettings().getUserAgentString(),
                        "",
                        path.contains("pdf") ? "application/pdf" : "video/mp4");
                    return true;
                }
                return false;
            }

            // External schemes like mailto, tel
            try {
                Intent intent = new Intent(Intent.ACTION_VIEW, uri);
                startActivity(intent);
                return true;
            } catch (Exception e) {
                return true;
            }
        }

        @Override
        public void onPageStarted(WebView view, String url, android.graphics.Bitmap favicon) {
            super.onPageStarted(view, url, favicon);
            errorLayout.setVisibility(View.GONE);
            webView.setVisibility(View.VISIBLE);
        }

        @Override
        public void onPageFinished(WebView view, String url) {
            super.onPageFinished(view, url);
            swipeRefreshLayout.setRefreshing(false);
            progressBar.setVisibility(View.GONE);

            // Smoothly fade out the branded splash screen
            if (splashLayout != null && splashLayout.getVisibility() == View.VISIBLE) {
                splashLayout.animate()
                        .alpha(0f)
                        .setDuration(400)
                        .withEndAction(() -> {
                            splashLayout.setVisibility(View.GONE);
                            splashLayout.setAlpha(1f);
                        });
            }

            // Inject the native speech recognition polyfill into the page
            injectSpeechPolyfill(view);
        }

        @Override
        public void onReceivedError(WebView view, WebResourceRequest request, WebResourceError error) {
            super.onReceivedError(view, request, error);
            if (request.isForMainFrame()) {
                if (splashLayout != null) {
                    splashLayout.setVisibility(View.GONE);
                }
                webView.setVisibility(View.GONE);
                errorLayout.setVisibility(View.VISIBLE);
                swipeRefreshLayout.setRefreshing(false);
                progressBar.setVisibility(View.GONE);
            }
        }
    }

    /**
     * Inject a JavaScript polyfill that replaces the missing Web Speech API
     * with a bridge to Android's native SpeechRecognizer. Also patches
     * window.open for PDF downloads to work inside WebView.
     */
    private void injectSpeechPolyfill(WebView view) {
        String polyfill = "(function() {" +
            // Prevent double injection
            "if (window._nativeSpeechInjected) return;" +
            "window._nativeSpeechInjected = true;" +

            // Callback handlers called from Java via callJs()
            "window._nativeSpeechOnStart = function(targetId, text) {" +
            "  var el = document.getElementById(targetId + '_timer');" +
            "  if (window._nativeSpeechTimerStart && window._nativeSpeechTimerStart[targetId]) return;" +
            "  if (!window._nativeSpeechTimerStart) window._nativeSpeechTimerStart = {};" +
            "  window._nativeSpeechTimerStart[targetId] = Date.now();" +
            "  var timerEls = {" +
            "    'audioTranscript': document.getElementById('audioSpeechTimer')," +
            "    'videoTranscript': document.getElementById('videoSpeechTimer')" +
            "  };" +
            "  var timerEl = timerEls[targetId];" +
            "  if (timerEl) {" +
            "    timerEl.textContent = 'Recording Time: 00:00';" +
            "    if (!window._nativeSpeechTimerInterval) window._nativeSpeechTimerInterval = {};" +
            "    if (window._nativeSpeechTimerInterval[targetId]) clearInterval(window._nativeSpeechTimerInterval[targetId]);" +
            "    window._nativeSpeechTimerInterval[targetId] = setInterval(function() {" +
            "      var elapsed = Math.floor((Date.now() - window._nativeSpeechTimerStart[targetId]) / 1000);" +
            "      var m = String(Math.floor(elapsed/60)).padStart(2,'0');" +
            "      var s = String(elapsed%60).padStart(2,'0');" +
            "      timerEl.textContent = 'Recording Time: ' + m + ':' + s;" +
            "    }, 1000);" +
            "  }" +
            "};" +

            "window._cleanSpeechText = function(text) {" +
            "  if (!text) return '';" +
            "  var words = text.trim().split(/\\s+/);" +
            "  var clean = [];" +
            "  for (var i = 0; i < words.length; i++) {" +
            "    var norm = words[i].toLowerCase().replace(/[^a-z0-9]/g, '');" +
            "    var prev = clean.length > 0 ? clean[clean.length - 1].toLowerCase().replace(/[^a-z0-9]/g, '') : '';" +
            "    if (norm && norm === prev) continue;" +
            "    clean.push(words[i]);" +
            "  }" +
            "  return clean.join(' ');" +
            "};" +
            "window._nativeSpeechOnResult = function(targetId, text) {" +
            "  var el = document.getElementById(targetId);" +
            "  if (el) el.value = window._cleanSpeechText(text);" +
            "};" +
            "window._nativeSpeechOnPartial = function(targetId, text) {" +
            "  var el = document.getElementById(targetId);" +
            "  if (el) el.value = window._cleanSpeechText(text);" +
            "};" +
            "window._nativeSpeechOnEnd = function(targetId, text) {" +
            "  if (window._nativeSpeechTimerInterval && window._nativeSpeechTimerInterval[targetId]) {" +
            "    clearInterval(window._nativeSpeechTimerInterval[targetId]);" +
            "    delete window._nativeSpeechTimerInterval[targetId];" +
            "  }" +
            "  if (window._nativeSpeechTimerStart) delete window._nativeSpeechTimerStart[targetId];" +
            "};" +

            // Route all speech through native bridge whenever AndroidSpeech is present
            "if (window.AndroidSpeech) {" +
            "  console.log('[AI Coach] Using native Android speech recognition');" +

            // Create a mock SpeechRecognition constructor so initSpeech() doesn't bail
            "  window._nativeSpeechTargets = {};" +

            "  window.SpeechRecognition = function() {" +
            "    this.continuous = true;" +
            "    this.interimResults = true;" +
            "    this.lang = 'en-US';" +
            "    this._targetId = null;" +
            "    this.onstart = null;" +
            "    this.onresult = null;" +
            "    this.onend = null;" +
            "    this.onerror = null;" +
            "    this.start = function() {" +
            "      if (this._targetId && window.AndroidSpeech) {" +
            "        window.AndroidSpeech.startListening(this._targetId);" +
            "      }" +
            "    };" +
            "    this.stop = function() {" +
            "      if (window.AndroidSpeech) window.AndroidSpeech.stopListening();" +
            "      if (typeof this.onend === 'function') this.onend();" +
            "      window._nativeSpeechOnEnd(this._targetId || '', '');" +
            "    };" +
            "    this.abort = this.stop;" +
            "  };" +

            // Re-define initSpeech to wire up native bridge
            "  var origInitSpeech = window.initSpeech;" +
            "  window.initSpeech = function(targetElementId, timerElementId) {" +
            "    var recognizer = new window.SpeechRecognition();" +
            "    recognizer._targetId = targetElementId;" +
            "    window._nativeSpeechTargets[targetElementId] = recognizer;" +
            "    return recognizer;" +
            "  };" +
            "}" +

            // Patch window.open for PDF downloads (the app uses window.open for /api/reports/pdf)
            "var origOpen = window.open;" +
            "window.open = function(url, target, features) {" +
            "  if (url && (url.indexOf('/api/reports/pdf') !== -1 || url.indexOf('/api/mock-interview/') !== -1)) {" +
            "    window.location.href = url;" +
            "    return null;" +
            "  }" +
            "  return origOpen ? origOpen.call(window, url, target, features) : null;" +
            "};" +

            "})();";

        view.evaluateJavascript(polyfill, null);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        stopListeningInternal();
        if (webView != null) {
            webView.removeJavascriptInterface("AndroidSpeech");
            webView.destroy();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        stopListeningInternal();
    }
}
