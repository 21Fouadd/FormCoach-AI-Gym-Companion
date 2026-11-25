import 'dart:async';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import '../models/models.dart';
import '../services/backend_pose_analyzer.dart';
import '../services/pose_analyzer.dart';
import '../theme/app_theme.dart';
import '../widgets/common.dart';
import 'session_summary_screen.dart';
import 'shell.dart' as shell show analyzer;

class LiveSessionScreen extends StatefulWidget {
  final Exercise exercise;

  /// Defaults to the app-wide analyzer from shell.dart. Tests inject a fake.
  final PoseAnalyzer? analyzer;

  const LiveSessionScreen({super.key, required this.exercise, this.analyzer});

  @override
  State<LiveSessionScreen> createState() => _LiveSessionScreenState();
}

class _LiveSessionScreenState extends State<LiveSessionScreen> {
  PoseAnalyzer get analyzer => widget.analyzer ?? shell.analyzer;
  CameraController? _camera;
  bool _cameraFailed = false;
  bool _recording = false;
  bool _paused = false;
  bool _started = false;

  StreamSubscription<FrameAnalysis>? _sub;
  Timer? _clock;
  Timer? _frameTimer;
  bool _sendingFrame = false;
  Duration _elapsed = Duration.zero;
  FrameAnalysis _frame = FrameAnalysis.empty;

  final List<double> _scores = [];
  final Map<String, int> _issueCounts = {};
  final Map<String, Severity> _issueSeverity = {};

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      final cams = await availableCameras();
      final cam = cams.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cams.first,
      );
      final controller =
          CameraController(cam, ResolutionPreset.high, enableAudio: false);
      await controller.initialize();
      if (!mounted) {
        await controller.dispose();
        return;
      }
      setState(() => _camera = controller);
    } catch (_) {
      if (mounted) setState(() => _cameraFailed = true);
    }
  }

  Future<void> _start() async {
    await analyzer.load();
    _sub = analyzer.analyzeLive(widget.exercise).listen(_onFrame);
    _clock = Timer.periodic(const Duration(seconds: 1), (_) {
      if (!_paused) setState(() => _elapsed += const Duration(seconds: 1));
    });
    if (analyzer is BackendPoseAnalyzer) {
      final backend = analyzer as BackendPoseAnalyzer;
      _frameTimer = Timer.periodic(const Duration(milliseconds: 260), (_) async {
        final c = _camera;
        if (!mounted ||
            _paused ||
            _recording ||
            _sendingFrame ||
            c == null ||
            !c.value.isInitialized) {
          return;
        }
        _sendingFrame = true;
        try {
          final xfile = await c.takePicture();
          final bytes = await xfile.readAsBytes();
          backend.sendFrameBytes(bytes);
        } catch (_) {
        } finally {
          _sendingFrame = false;
        }
      });
    }
    setState(() => _started = true);
  }

  void _onFrame(FrameAnalysis f) {
    if (_paused) return;
    _scores.add(f.formScore);
    for (final i in f.issues) {
      if (i.severity == Severity.good) continue;
      _issueCounts[i.label] = (_issueCounts[i.label] ?? 0) + 1;
      _issueSeverity[i.label] = i.severity;
    }
    setState(() => _frame = f);
  }

  Future<void> _toggleRecord() async {
    final c = _camera;
    if (c == null) return;
    if (_recording) return;
    try {
      await c.startVideoRecording();
      setState(() => _recording = true);
    } catch (_) {}
  }

  Future<void> _finish() async {
    String? path;
    final c = _camera;
    if (_recording && c != null) {
      try {
        final file = await c.stopVideoRecording();
        path = file.path;
      } catch (_) {}
    }
    _frameTimer?.cancel();
    if (analyzer is BackendPoseAnalyzer) {
      await (analyzer as BackendPoseAnalyzer).finishLiveSet();
    }
    await _sub?.cancel();
    _clock?.cancel();

    final avg = _scores.isEmpty
        ? 0.0
        : _scores.reduce((a, b) => a + b) / _scores.length;
    final top = (_issueCounts.entries.toList()
          ..sort((a, b) => b.value.compareTo(a.value)))
        .take(3)
        .map((e) => FormIssue(e.key, _issueSeverity[e.key]!))
        .toList();

    if (!mounted) return;
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(
        builder: (_) => SessionSummaryScreen(
          result: SessionResult(
            exercise: widget.exercise,
            reps: _frame.reps,
            avgForm: avg,
            duration: _elapsed,
            topIssues: top,
            videoPath: path,
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _frameTimer?.cancel();
    _sub?.cancel();
    _clock?.cancel();
    _camera?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          _cameraLayer(),
          if (_started)
            CustomPaint(painter: SkeletonPainter(_frame.landmarks)),
          // Top and bottom gradients keep the HUD readable on any background.
          const Positioned(
            top: 0,
            left: 0,
            right: 0,
            height: 160,
            child: DecoratedBox(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [Color(0xCC000000), Color(0x00000000)],
                ),
              ),
            ),
          ),
          const Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            height: 300,
            child: DecoratedBox(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.bottomCenter,
                  end: Alignment.topCenter,
                  colors: [Color(0xE6000000), Color(0x00000000)],
                ),
              ),
            ),
          ),
          SafeArea(child: _hud()),
        ],
      ),
    );
  }

  Widget _cameraLayer() {
    final c = _camera;
    if (c != null && c.value.isInitialized) {
      return FittedBox(
        fit: BoxFit.cover,
        child: SizedBox(
          width: c.value.previewSize?.height ?? 1,
          height: c.value.previewSize?.width ?? 1,
          child: CameraPreview(c),
        ),
      );
    }
    return Container(
      color: AppColors.bg,
      alignment: Alignment.center,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            _cameraFailed ? Icons.videocam_off_outlined : Icons.videocam_outlined,
            size: 48,
            color: AppColors.textDim,
          ),
          const SizedBox(height: 10),
          Text(
            _cameraFailed ? 'Camera unavailable' : 'Starting camera...',
            style: const TextStyle(color: AppColors.textDim),
          ),
        ],
      ),
    );
  }

  Widget _hud() {
    return Padding(
      padding: const EdgeInsets.all(AppSpace.md),
      child: Column(
        children: [
          // Top bar
          Row(
            children: [
              _RoundButton(
                icon: Icons.close,
                onTap: () => Navigator.pop(context),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      widget.exercise.id == 'auto' &&
                              _frame.detectedExercise != null &&
                              _frame.detectedExercise != 'auto'
                          ? 'Auto: ${_frame.detectedExercise!.toUpperCase()}'
                          : widget.exercise.name,
                      style: const TextStyle(
                          fontSize: 18, fontWeight: FontWeight.w800),
                    ),
                    Text(fmtDuration(_elapsed),
                        style: const TextStyle(
                            color: AppColors.textDim, fontSize: 13)),
                  ],
                ),
              ),
              if (_frame.athleteName != null)
                Container(
                  margin: const EdgeInsets.only(right: 8),
                  padding:
                      const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                  decoration: BoxDecoration(
                    color: AppColors.surface.withValues(alpha: 0.9),
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(color: AppColors.accent),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.end,
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        _frame.athleteName!,
                        style: const TextStyle(
                          color: AppColors.accent,
                          fontSize: 12,
                          fontWeight: FontWeight.w800,
                        ),
                      ),
                      if (_frame.athleteMeta != null)
                        Text(
                          _frame.athleteMeta!,
                          style: const TextStyle(
                            color: AppColors.text,
                            fontSize: 10,
                          ),
                        ),
                    ],
                  ),
                ),
              if (_recording)
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                  decoration: BoxDecoration(
                    color: AppColors.bad,
                    borderRadius: BorderRadius.circular(999),
                  ),
                  child: const Row(
                    children: [
                      Icon(Icons.fiber_manual_record, size: 12),
                      SizedBox(width: 4),
                      Text('REC',
                          style: TextStyle(
                              fontSize: 12, fontWeight: FontWeight.w800)),
                    ],
                  ),
                ),
            ],
          ),
          const Spacer(),

          if (!_started) ...[
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.black.withValues(alpha: 0.6),
                borderRadius: BorderRadius.circular(AppRadius.lg),
              ),
              child: Text(widget.exercise.cue,
                  textAlign: TextAlign.center,
                  style: const TextStyle(fontSize: 15)),
            ),
            const SizedBox(height: AppSpace.md),
            FilledButton.icon(
              onPressed: _start,
              icon: const Icon(Icons.play_arrow),
              label: const Text('Start set'),
            ),
          ] else ...[
            // Rep counter + score
            Row(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('REPS',
                          style: TextStyle(
                              color: AppColors.textDim,
                              letterSpacing: 1.4,
                              fontWeight: FontWeight.w700)),
                      Text('${_frame.reps}',
                          style: const TextStyle(
                              fontSize: 96,
                              height: 1,
                              fontWeight: FontWeight.w900)),
                      const SizedBox(height: 6),
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 10, vertical: 4),
                        decoration: BoxDecoration(
                          color: AppColors.accent,
                          borderRadius: BorderRadius.circular(6),
                        ),
                        child: Text(_frame.phase.toUpperCase(),
                            style: const TextStyle(
                                color: AppColors.onAccent,
                                fontWeight: FontWeight.w800,
                                fontSize: 12,
                                letterSpacing: 1)),
                      ),
                    ],
                  ),
                ),
                ScoreRing(score: _frame.formScore, size: 92, stroke: 9),
              ],
            ),
            const SizedBox(height: AppSpace.md),
            // Live feedback
            Align(
              alignment: Alignment.centerLeft,
              child: Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [for (final i in _frame.issues) SeverityChip(i)],
              ),
            ),
            const SizedBox(height: AppSpace.md),
            Row(
              children: [
                _RoundButton(
                  icon: _recording
                      ? Icons.fiber_manual_record
                      : Icons.videocam_outlined,
                  color: _recording ? AppColors.bad : null,
                  onTap: _toggleRecord,
                  size: 56,
                ),
                const SizedBox(width: 12),
                _RoundButton(
                  icon: _paused ? Icons.play_arrow : Icons.pause,
                  onTap: () => setState(() => _paused = !_paused),
                  size: 56,
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: FilledButton(
                    onPressed: _finish,
                    child: const Text('Finish set'),
                  ),
                ),
              ],
            ),
          ],
        ],
      ),
    );
  }
}

class _RoundButton extends StatelessWidget {
  final IconData icon;
  final VoidCallback onTap;
  final double size;
  final Color? color;

  const _RoundButton({
    required this.icon,
    required this.onTap,
    this.size = 44,
    this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.black.withValues(alpha: 0.55),
      shape: const CircleBorder(),
      child: InkWell(
        customBorder: const CircleBorder(),
        onTap: onTap,
        child: SizedBox(
          width: size,
          height: size,
          child: Icon(icon, color: color ?? AppColors.text),
        ),
      ),
    );
  }
}
