import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

/// Shared connection & Face ID configuration for the FastAPI AI backend.
class BackendConfig {
  static const String _keyUrl = 'fc_backend_url';
  static const String _keyVoice = 'fc_voice_cues';

  /// Default points to laptop on local Wi-Fi; switch to http://10.0.2.2:8000
  /// in Profile when running inside an Android Emulator.
  static String baseUrl = 'http://192.168.1.3:8000';
  static bool voiceEnabled = true;

  static Future<void> load() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final savedUrl = prefs.getString(_keyUrl);
      if (savedUrl != null && savedUrl.trim().isNotEmpty) {
        baseUrl = savedUrl.trim();
      }
      voiceEnabled = prefs.getBool(_keyVoice) ?? true;
    } catch (_) {}
  }

  static Future<void> setBaseUrl(String url) async {
    final cleaned = url.trim().replaceAll(RegExp(r'/$'), '');
    if (cleaned.isEmpty) return;
    baseUrl = cleaned;
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(_keyUrl, cleaned);
    } catch (_) {}
  }

  static Future<void> setVoiceEnabled(bool enabled) async {
    voiceEnabled = enabled;
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setBool(_keyVoice, enabled);
    } catch (_) {}
  }

  static String wsLiveUrl(String exercise) {
    final clean = baseUrl.replaceAll(RegExp(r'/$'), '');
    final wsBase = clean.startsWith('https://')
        ? clean.replaceFirst('https://', 'wss://')
        : clean.replaceFirst('http://', 'ws://');
    return '$wsBase/ws/live/$exercise';
  }

  static Future<Map<String, dynamic>> checkHealth() async {
    final res = await http
        .get(Uri.parse('$baseUrl/api/health'))
        .timeout(const Duration(seconds: 6));
    if (res.statusCode != 200) {
      throw Exception('Backend returned HTTP ${res.statusCode}');
    }
    return jsonDecode(res.body) as Map<String, dynamic>;
  }

  static Future<Map<String, dynamic>> registerFaceProfile({
    required String name,
    required int age,
    required double heightCm,
    required double weightKg,
    String fitnessLevel = 'Auto',
    required String photoPath,
  }) async {
    final req = http.MultipartRequest(
      'POST',
      Uri.parse('$baseUrl/api/users/register'),
    )
      ..fields['name'] = name.trim()
      ..fields['age'] = age.toString()
      ..fields['height_cm'] = heightCm.toString()
      ..fields['weight_kg'] = weightKg.toString()
      ..fields['fitness_level'] = fitnessLevel
      ..files.add(await http.MultipartFile.fromPath('photo', photoPath));

    final streamed = await req.send().timeout(const Duration(seconds: 20));
    final body = await streamed.stream.bytesToString();
    final json = jsonDecode(body) as Map<String, dynamic>;
    if (streamed.statusCode != 200) {
      throw Exception(json['detail'] ?? 'Face ID enrollment failed');
    }
    return json;
  }
}
