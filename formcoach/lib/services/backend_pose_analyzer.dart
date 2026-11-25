import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui';

import 'package:flutter_tts/flutter_tts.dart';
import 'package:http/http.dart' as http;
import 'package:web_socket_channel/web_socket_channel.dart';

import '../models/models.dart';
import 'backend_config.dart';
import 'pose_analyzer.dart';

/// Connects FormCoach directly to the FastAPI AI backend (`pose_engine.py`,
/// `face_id.py`, `profile_ai.py`), including Stage-1 Auto Exercise Detection,
/// 12-joint MediaPipe skeleton streaming, Face ID athlete HUD, and TTS voice cues.
class BackendPoseAnalyzer implements PoseAnalyzer {
  static const List<int> _coreJointIds = [
    11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28,
  ];

  final FlutterTts _tts = FlutterTts();
  WebSocketChannel? _channel;
  StreamSubscription? _wsSub;
  StreamController<FrameAnalysis>? _liveController;
  DateTime _lastSpokenAt = DateTime.fromMillisecondsSinceEpoch(0);
  bool _ttsReady = false;

  /// Populated after [analyzeVideo] finishes when the backend rendered an annotated MP4.
  String? lastAnnotatedVideoUrl;

  /// Raw summary payload from the latest [analyzeVideo] call.
  Map<String, dynamic>? lastSummary;

  static String mapExerciseId(Exercise exercise) {
    switch (exercise.id) {
      case 'auto':
        return 'auto';
      case 'squat':
        return 'squat';
      case 'pushup':
        return 'push';
      case 'ohp':
        return 'shoulder';
      case 'lunge':
        return 'lunge';
      case 'curl':
        return 'curl';
      case 'deadlift':
      case 'row':
        return 'row';
      case 'arm_abduction':
        return 'arm_abduction';
      default:
        return 'auto';
    }
  }

  @override
  Future<void> load() async {
    await BackendConfig.load();
    if (!_ttsReady) {
      try {
        await _tts.setLanguage('en-US');
        await _tts.setSpeechRate(0.5);
        _ttsReady = true;
      } catch (_) {}
    }
  }

  Future<void> _speak(String? text) async {
    if (!BackendConfig.voiceEnabled || text == null || text.trim().isEmpty) {
      return;
    }
    final now = DateTime.now();
    if (now.difference(_lastSpokenAt).inMilliseconds < 2200) return;
    _lastSpokenAt = now;
    try {
      await _tts.stop();
      await _tts.speak(text.trim());
    } catch (_) {}
  }

  @override
  Stream<FrameAnalysis> analyzeLive(Exercise exercise) {
    _closeLive();
    final controller = StreamController<FrameAnalysis>.broadcast();
    _liveController = controller;

    final ex = mapExerciseId(exercise);
    try {
      final channel = WebSocketChannel.connect(
        Uri.parse(BackendConfig.wsLiveUrl(ex)),
      );
      _channel = channel;

      _wsSub = channel.stream.listen(
        (raw) {
          try {
            final data = jsonDecode(raw as String) as Map<String, dynamic>;
            if (data['type'] != 'frame_result') return;

            final verdict = (data['verdict'] as String?) ?? 'Ready';
            final stage = (data['stage'] as String?) ?? 'up';
            final cues = List<String>.from(data['cues'] ?? const []);
            final isBad = verdict.toLowerCase() == 'incorrect';

            final issues = <FormIssue>[
              for (final c in cues.take(2))
                FormIssue(
                  c,
                  isBad
                      ? Severity.bad
                      : (verdict == 'Ready' || verdict == 'Detecting'
                          ? Severity.warn
                          : Severity.good),
                ),
            ];

            final rawLms = data['landmarks'];
            final landmarks = List<Offset>.filled(
              _coreJointIds.length,
              const Offset(-1, -1),
            );
            if (rawLms is Map) {
              for (var i = 0; i < _coreJointIds.length; i++) {
                final pt = rawLms[_coreJointIds[i].toString()];
                if (pt is List && pt.length >= 2) {
                  landmarks[i] = Offset(
                    (pt[0] as num).toDouble(),
                    (pt[1] as num).toDouble(),
                  );
                }
              }
            }

            String? athleteName;
            String? athleteMeta;
            final user = data['user'];
            if (user is Map) {
              final conf =
                  ((data['recognition_confidence'] as num?)?.toDouble() ?? 0.0) *
                      100.0;
              athleteName = conf > 0
                  ? '${user['name']} (${conf.round()}% ID)'
                  : '${user['name']}';
              athleteMeta =
                  'Age ${user['age']} • ${(user['height_cm'] as num).round()} cm • ${user['fitness_level']}';
            }

            List<double>? personBox;
            if (data['person_box'] is List) {
              personBox = (data['person_box'] as List)
                  .map((e) => (e as num).toDouble())
                  .toList();
            }

            final spoken = data['spoken_cue'] as String?;
            if (spoken != null && spoken.isNotEmpty) {
              _speak(spoken);
            }

            if (!controller.isClosed) {
              controller.add(FrameAnalysis(
                reps: (data['rep_count'] as num?)?.toInt() ?? 0,
                formScore:
                    (data['form_score'] as num?)?.toDouble() ?? 100.0,
                phase: stage,
                issues: issues,
                landmarks: landmarks,
                athleteName: athleteName,
                athleteMeta: athleteMeta,
                personBox: personBox,
                detectedExercise: data['exercise'] as String?,
              ));
            }
          } catch (_) {}
        },
        onError: (_) {
          if (!controller.isClosed) {
            controller.add(const FrameAnalysis(
              reps: 0,
              formScore: 0,
              phase: 'Offline',
              issues: [
                FormIssue(
                  'Backend unreachable — check server URL in Profile',
                  Severity.warn,
                ),
              ],
            ));
          }
        },
      );
    } catch (_) {
      controller.add(const FrameAnalysis(
        reps: 0,
        formScore: 0,
        phase: 'Offline',
        issues: [
          FormIssue(
            'Could not connect to AI server — update URL in Profile',
            Severity.warn,
          ),
        ],
      ));
    }

    controller.onCancel = _closeLive;
    return controller.stream;
  }

  /// Sends a JPEG camera frame over the active WebSocket connection.
  void sendFrameBytes(Uint8List jpegBytes) {
    final ch = _channel;
    if (ch == null || jpegBytes.isEmpty) return;
    try {
      ch.sink.add(jsonEncode({'frame_b64': base64Encode(jpegBytes)}));
    } catch (_) {}
  }

  /// Signals the backend to save the live workout set to SQLite.
  Future<void> finishLiveSet() async {
    try {
      _channel?.sink.add(jsonEncode({'action': 'finish'}));
      await Future.delayed(const Duration(milliseconds: 250));
    } catch (_) {}
    await _closeLive();
  }

  Future<void> _closeLive() async {
    await _wsSub?.cancel();
    _wsSub = null;
    try {
      await _channel?.sink.close();
    } catch (_) {}
    _channel = null;
    _liveController = null;
  }

  @override
  Future<List<VideoFinding>> analyzeVideo(String path, Exercise exercise) async {
    await load();
    lastAnnotatedVideoUrl = null;
    lastSummary = null;

    final ex = mapExerciseId(exercise);
    try {
      final req = http.MultipartRequest(
        'POST',
        Uri.parse('${BackendConfig.baseUrl}/api/analyze/video'),
      )
        ..fields['exercise'] = ex
        ..files.add(await http.MultipartFile.fromPath('video', path));

      final streamed = await req.send().timeout(const Duration(seconds: 90));
      final body = await streamed.stream.bytesToString();
      if (streamed.statusCode != 200) {
        throw Exception('HTTP ${streamed.statusCode}: $body');
      }

      final data = jsonDecode(body) as Map<String, dynamic>;
      lastSummary = data;
      final videoUrl = data['video_url'] as String?;
      if (videoUrl != null && videoUrl.isNotEmpty) {
        lastAnnotatedVideoUrl = '${BackendConfig.baseUrl}$videoUrl';
      }

      final summaryCues = List<String>.from(data['summary_cues'] ?? const []);
      if (summaryCues.isNotEmpty) {
        _speak(
          'Analyzed ${data['total_reps'] ?? 0} reps. ${summaryCues.first}',
        );
      }

      final reps = List<Map<String, dynamic>>.from(data['reps'] ?? const []);
      if (reps.isEmpty) {
        return const [
          VideoFinding(
            at: Duration.zero,
            rep: 1,
            title: 'No complete reps detected',
            detail:
                'Ensure your full body is visible from a side or 45-degree angle.',
            severity: Severity.warn,
          ),
        ];
      }

      return [
        for (final r in reps)
          VideoFinding(
            at: Duration(
              milliseconds: (((r['start_sec'] as num?)?.toDouble() ?? 0.0) *
                      1000)
                  .round(),
            ),
            rep: (r['rep'] as num?)?.toInt() ?? 1,
            title: (r['is_incorrect'] == true) ? 'Needs form fix' : 'Clean rep',
            detail: List<String>.from(r['cues'] ?? const []).join(' • '),
            severity: (r['is_incorrect'] == true) ? Severity.bad : Severity.good,
          ),
      ];
    } catch (e) {
      return [
        VideoFinding(
          at: Duration.zero,
          rep: 1,
          title: 'Backend unavailable',
          detail:
              'Check server URL (${BackendConfig.baseUrl}) in Profile -> $e',
          severity: Severity.warn,
        ),
      ];
    }
  }

  @override
  void dispose() {
    _closeLive();
    _tts.stop();
  }
}
