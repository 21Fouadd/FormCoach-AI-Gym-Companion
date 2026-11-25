import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:image_picker/image_picker.dart';
import 'package:video_player/video_player.dart';
import '../services/api_service.dart';

class PostSetVideoScreen extends StatefulWidget {
  const PostSetVideoScreen({super.key});

  @override
  State<PostSetVideoScreen> createState() => _PostSetVideoScreenState();
}

class _PostSetVideoScreenState extends State<PostSetVideoScreen> {
  final FlutterTts _tts = FlutterTts();
  String _exercise = 'squat';
  bool _loading = false;
  Map<String, dynamic>? _result;
  VideoPlayerController? _videoCtrl;

  Future<void> _pickAndAnalyze() async {
    final picked = await ImagePicker().pickVideo(source: ImageSource.gallery);
    if (picked == null) return;

    setState(() {
      _loading = true;
      _result = null;
    });
    try {
      final res = await ApiService.analyzePostSetVideo(
        exercise: _exercise,
        videoFile: File(picked.path),
      );
      final videoUrl = '${ApiService.baseUrl}${res['video_url']}';
      await _videoCtrl?.dispose();
      _videoCtrl = VideoPlayerController.networkUrl(Uri.parse(videoUrl));
      await _videoCtrl!.initialize();
      await _videoCtrl!.setLooping(true);
      await _videoCtrl!.play();

      final cues = List<String>.from(res['summary_cues'] ?? []);
      if (cues.isNotEmpty) {
        _tts.speak('You completed ${res['total_reps']} reps. ${cues.first}');
      }
      if (mounted) setState(() => _result = res);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: $e')));
      }
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  void dispose() {
    _videoCtrl?.dispose();
    _tts.stop();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final reps = List<Map<String, dynamic>>.from(_result?['reps'] ?? []);
    final user = _result?['recognized_user'] as Map<String, dynamic>?;

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Row(
          children: [
            DropdownButton<String>(
              value: _exercise,
              items: const [
                DropdownMenuItem(value: 'squat', child: Text('Squat')),
                DropdownMenuItem(value: 'push', child: Text('Push-Up')),
                DropdownMenuItem(value: 'curl', child: Text('Bicep Curl')),
                DropdownMenuItem(value: 'shoulder', child: Text('Shoulder Press')),
                DropdownMenuItem(value: 'row', child: Text('Bent-Over Row')),
              ],
              onChanged: (v) => setState(() => _exercise = v ?? 'squat'),
            ),
            const Spacer(),
            FilledButton.icon(
              onPressed: _loading ? null : _pickAndAnalyze,
              icon: const Icon(Icons.video_library),
              label: Text(_loading ? 'Analyzing...' : 'Pick Set Video'),
            ),
          ],
        ),
        if (_loading) const Padding(padding: EdgeInsets.all(24), child: Center(child: CircularProgressIndicator())),
        if (_videoCtrl != null && _videoCtrl!.value.isInitialized) ...[
          const SizedBox(height: 14),
          AspectRatio(aspectRatio: _videoCtrl!.value.aspectRatio, child: VideoPlayer(_videoCtrl!)),
        ],
        if (_result != null) ...[
          const SizedBox(height: 14),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(14),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    user != null
                        ? '${user['name']} • Age ${user['age']} • ${user['height_cm']} cm • ${user['fitness_level']}'
                        : 'Athlete: Unrecognized (register face in Athletes tab)',
                    style: const TextStyle(fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 6),
                  Text('Total Reps: ${_result!['total_reps']}  |  Correct: ${_result!['correct_reps']}  |  Score: ${_result!['form_score']}%'),
                  const SizedBox(height: 6),
                  Text('Cues: ${(List<String>.from(_result!['summary_cues'] ?? [])).join(" • ")}'),
                ],
              ),
            ),
          ),
          const SizedBox(height: 8),
          for (final r in reps)
            Card(
              child: ListTile(
                title: Text('Rep #${r['rep']} (${r['start_sec']}s – ${r['end_sec']}s)'),
                subtitle: Text((List<String>.from(r['cues'] ?? [])).join(', ')),
                trailing: Text(
                  r['verdict'].toString().toUpperCase(),
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    color: r['is_incorrect'] == true ? Colors.redAccent : Colors.greenAccent,
                  ),
                ),
              ),
            ),
        ],
      ],
    );
  }
}
