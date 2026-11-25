import 'dart:async';
import 'dart:convert';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import '../services/api_service.dart';
import '../widgets/hud_overlay_painter.dart';

class LiveTrainerScreen extends StatefulWidget {
  const LiveTrainerScreen({super.key});

  @override
  State<LiveTrainerScreen> createState() => _LiveTrainerScreenState();
}

class _LiveTrainerScreenState extends State<LiveTrainerScreen> {
  CameraController? _cameraController;
  WebSocketChannel? _channel;
  Timer? _frameTimer;
  final FlutterTts _tts = FlutterTts();

  String _exercise = 'squat';
  bool _running = false;
  bool _voiceEnabled = true;
  bool _busyFrame = false;

  int _repCount = 0;
  double _formScore = 100.0;
  String _verdict = 'Ready';
  String _cue = 'Tap Start and face camera for Face ID.';
  List<double>? _personBox;
  Map<String, dynamic> _landmarks = {};
  Map<String, dynamic>? _recognizedUser;
  double _confidence = 0.0;

  @override
  void initState() {
    super.initState();
    _initCamera();
    _tts.setSpeechRate(0.52);
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) return;
      final selected = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );
      _cameraController = CameraController(selected, ResolutionPreset.medium, enableAudio: false);
      await _cameraController!.initialize();
      if (mounted) setState(() {});
    } catch (_) {}
  }

  Future<void> _startSession() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) return;
    _channel = WebSocketChannel.connect(Uri.parse(ApiService.wsLiveUrl(_exercise)));
    _channel!.stream.listen((raw) {
      final data = jsonDecode(raw as String) as Map<String, dynamic>;
      if (data['type'] == 'frame_result' && mounted) {
        final spoken = data['spoken_cue'] as String?;
        if (spoken != null && spoken.isNotEmpty && _voiceEnabled) {
          _tts.speak(spoken);
        }
        setState(() {
          _repCount = (data['rep_count'] as num?)?.toInt() ?? _repCount;
          _formScore = (data['form_score'] as num?)?.toDouble() ?? _formScore;
          _verdict = (data['verdict'] as String?) ?? _verdict;
          final cues = List<String>.from(data['cues'] ?? []);
          if (cues.isNotEmpty) _cue = cues.first;
          _personBox = data['person_box'] != null
              ? List<double>.from((data['person_box'] as List).map((e) => (e as num).toDouble()))
              : null;
          _landmarks = Map<String, dynamic>.from(data['landmarks'] ?? {});
          _recognizedUser = data['user'] != null ? Map<String, dynamic>.from(data['user']) : null;
          _confidence = (data['recognition_confidence'] as num?)?.toDouble() ?? 0.0;
        });
      }
    });

    setState(() => _running = true);
    _frameTimer = Timer.periodic(const Duration(milliseconds: 220), (_) async {
      if (!_running || _busyFrame || _cameraController == null) return;
      _busyFrame = true;
      try {
        final xfile = await _cameraController!.takePicture();
        final bytes = await xfile.readAsBytes();
        _channel?.sink.add(jsonEncode({'frame_b64': base64Encode(bytes)}));
      } catch (_) {
      } finally {
        _busyFrame = false;
      }
    });
  }

  Future<void> _stopSession() async {
    _frameTimer?.cancel();
    _channel?.sink.add(jsonEncode({'action': 'finish'}));
    await Future.delayed(const Duration(milliseconds: 300));
    await _channel?.sink.close();
    if (mounted) setState(() => _running = false);
  }

  @override
  void dispose() {
    _frameTimer?.cancel();
    _channel?.sink.close();
    _cameraController?.dispose();
    _tts.stop();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isBad = _verdict.toLowerCase() == 'incorrect';
    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          child: Row(
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
                onChanged: _running ? null : (v) => setState(() => _exercise = v ?? 'squat'),
              ),
              const Spacer(),
              IconButton(
                icon: Icon(_voiceEnabled ? Icons.volume_up : Icons.volume_off),
                onPressed: () => setState(() => _voiceEnabled = !_voiceEnabled),
              ),
              FilledButton(
                style: FilledButton.styleFrom(backgroundColor: _running ? Colors.redAccent : Colors.tealAccent),
                onPressed: _running ? _stopSession : _startSession,
                child: Text(_running ? 'Stop & Save' : 'Start Live', style: const TextStyle(color: Colors.black)),
              ),
            ],
          ),
        ),
        Expanded(
          child: Stack(
            fit: StackFit.expand,
            children: [
              if (_cameraController != null && _cameraController!.value.isInitialized)
                CameraPreview(_cameraController!)
              else
                const Center(child: Text('Initializing camera...')),
              CustomPaint(
                painter: HudOverlayPainter(
                  personBox: _personBox,
                  landmarks: _landmarks,
                  user: _recognizedUser,
                  recognitionConfidence: _confidence,
                  verdict: _verdict,
                ),
              ),
            ],
          ),
        ),
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(14),
          color: isBad ? const Color(0xFF3A1216) : const Color(0xFF10251D),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text('Reps: $_repCount', style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                  Text('Form Score: ${_formScore.toStringAsFixed(0)}%', style: const TextStyle(fontSize: 16)),
                  Text(_verdict.toUpperCase(), style: TextStyle(color: isBad ? Colors.redAccent : Colors.greenAccent, fontWeight: FontWeight.bold)),
                ],
              ),
              const SizedBox(height: 6),
              Text('Coach: $_cue', style: const TextStyle(fontSize: 15)),
            ],
          ),
        ),
      ],
    );
  }
}
