import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class ApiService {
  /// Default for Android Emulator -> host PC localhost.
  /// On a real phone on the same Wi-Fi, tap the Server Icon in the AppBar
  /// and enter e.g. http://192.168.1.15:8000
  static String baseUrl = 'http://10.0.2.2:8000';

  static String wsLiveUrl(String exercise) {
    final clean = baseUrl.replaceAll(RegExp(r'/$'), '');
    final wsBase = clean.startsWith('https://')
        ? clean.replaceFirst('https://', 'wss://')
        : clean.replaceFirst('http://', 'ws://');
    return '$wsBase/ws/live/$exercise';
  }

  static Future<Map<String, dynamic>> getHealth() async {
    final res = await http.get(Uri.parse('$baseUrl/api/health'));
    if (res.statusCode != 200) {
      throw Exception('Backend unreachable (${res.statusCode})');
    }
    return jsonDecode(res.body) as Map<String, dynamic>;
  }

  static Future<List<Map<String, dynamic>>> listUsers() async {
    final res = await http.get(Uri.parse('$baseUrl/api/users'));
    if (res.statusCode != 200) {
      throw Exception('Failed to load users (${res.statusCode})');
    }
    final data = jsonDecode(res.body) as Map<String, dynamic>;
    return List<Map<String, dynamic>>.from(data['users'] ?? []);
  }

  static Future<Map<String, dynamic>> registerUser({
    required String name,
    required int age,
    required double heightCm,
    required double weightKg,
    required String fitnessLevel,
    required File photoFile,
  }) async {
    final req = http.MultipartRequest('POST', Uri.parse('$baseUrl/api/users/register'))
      ..fields['name'] = name
      ..fields['age'] = age.toString()
      ..fields['height_cm'] = heightCm.toString()
      ..fields['weight_kg'] = weightKg.toString()
      ..fields['fitness_level'] = fitnessLevel
      ..files.add(await http.MultipartFile.fromPath('photo', photoFile.path));

    final streamed = await req.send();
    final body = await streamed.stream.bytesToString();
    final json = jsonDecode(body) as Map<String, dynamic>;
    if (streamed.statusCode != 200) {
      throw Exception(json['detail'] ?? 'Registration failed');
    }
    return json;
  }

  static Future<void> deleteUser(int userId) async {
    final res = await http.delete(Uri.parse('$baseUrl/api/users/$userId'));
    if (res.statusCode != 200) {
      throw Exception('Failed to delete user');
    }
  }

  static Future<Map<String, dynamic>> analyzePostSetVideo({
    required String exercise,
    required File videoFile,
  }) async {
    final req = http.MultipartRequest('POST', Uri.parse('$baseUrl/api/analyze/video'))
      ..fields['exercise'] = exercise
      ..files.add(await http.MultipartFile.fromPath('video', videoFile.path));

    final streamed = await req.send();
    final body = await streamed.stream.bytesToString();
    final json = jsonDecode(body) as Map<String, dynamic>;
    if (streamed.statusCode != 200) {
      throw Exception(json['detail'] ?? 'Video analysis failed');
    }
    return json;
  }
}
