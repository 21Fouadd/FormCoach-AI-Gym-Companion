import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:video_player/video_player.dart';
import '../data/mock_data.dart';
import '../models/models.dart';
import '../services/backend_pose_analyzer.dart';
import '../theme/app_theme.dart';
import '../widgets/common.dart';
import 'shell.dart' show analyzer;

class VideoAnalysisScreen extends StatefulWidget {
  final String? initialPath;
  final Exercise? initialExercise;
  const VideoAnalysisScreen({super.key, this.initialPath, this.initialExercise});

  @override
  State<VideoAnalysisScreen> createState() => _VideoAnalysisScreenState();
}

class _VideoAnalysisScreenState extends State<VideoAnalysisScreen> {
  final _picker = ImagePicker();
  VideoPlayerController? _video;
  late Exercise _exercise = widget.initialExercise ?? exercises.first;
  List<VideoFinding>? _findings;
  bool _analyzing = false;

  @override
  void initState() {
    super.initState();
    if (widget.initialPath != null) _load(widget.initialPath!);
  }

  Future<void> _pick(ImageSource source) async {
    final file = await _picker.pickVideo(source: source);
    if (file != null) await _load(file.path);
  }

  Future<void> _load(String path) async {
    await _video?.dispose();
    final c = VideoPlayerController.file(File(path));
    await c.initialize();
    c.addListener(() {
      if (mounted) setState(() {});
    });
    setState(() {
      _video = c;
      _findings = null;
    });
    _analyze(path);
  }

  Future<void> _analyze(String path) async {
    setState(() => _analyzing = true);
    final result = await analyzer.analyzeVideo(path, _exercise);
    if (!mounted) return;
    if (analyzer is BackendPoseAnalyzer) {
      final url = (analyzer as BackendPoseAnalyzer).lastAnnotatedVideoUrl;
      if (url != null) {
        try {
          final remote = VideoPlayerController.networkUrl(Uri.parse(url));
          await remote.initialize();
          remote.addListener(() {
            if (mounted) setState(() {});
          });
          await _video?.dispose();
          if (mounted) {
            _video = remote;
          } else {
            await remote.dispose();
          }
        } catch (_) {}
      }
    }
    if (!mounted) return;
    setState(() {
      _findings = result;
      _analyzing = false;
    });
  }

  @override
  void dispose() {
    _video?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final v = _video;
    return ListView(
      padding: const EdgeInsets.fromLTRB(
          AppSpace.md, AppSpace.sm, AppSpace.md, AppSpace.xl),
      children: [
        const Text('Video form check',
            style: TextStyle(fontSize: 28, fontWeight: FontWeight.w800)),
        const SizedBox(height: 4),
        const Text('Record or upload a set and get rep-by-rep feedback.',
            style: TextStyle(color: AppColors.textDim)),
        const SizedBox(height: AppSpace.md),

        DropdownButtonFormField<Exercise>(
          value: _exercise,
          dropdownColor: AppColors.surfaceHi,
          decoration: InputDecoration(
            filled: true,
            fillColor: AppColors.surface,
            labelText: 'Exercise',
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(AppRadius.md),
              borderSide: BorderSide.none,
            ),
          ),
          items: [
            for (final e in exercises)
              DropdownMenuItem(value: e, child: Text(e.name)),
          ],
          onChanged: (e) => setState(() => _exercise = e!),
        ),
        const SizedBox(height: AppSpace.md),

        if (v == null) _emptyState() else _player(v),

        if (_analyzing) ...[
          const SizedBox(height: AppSpace.lg),
          const Center(child: CircularProgressIndicator(color: AppColors.accent)),
          const SizedBox(height: 10),
          const Center(
              child: Text('Analyzing your form...',
                  style: TextStyle(color: AppColors.textDim))),
        ],

        if (_findings != null && v != null) ..._results(v, _findings!),
      ],
    );
  }

  Widget _emptyState() {
    return Panel(
      padding: const EdgeInsets.all(24),
      child: Column(
        children: [
          const Icon(Icons.video_library_outlined,
              size: 44, color: AppColors.accent),
          const SizedBox(height: 12),
          const Text('No video yet',
              style: TextStyle(fontSize: 17, fontWeight: FontWeight.w700)),
          const SizedBox(height: 4),
          const Text('Film from the side or front with your whole body in frame.',
              textAlign: TextAlign.center,
              style: TextStyle(color: AppColors.textDim)),
          const SizedBox(height: 20),
          FilledButton.icon(
            onPressed: () => _pick(ImageSource.camera),
            icon: const Icon(Icons.videocam),
            label: const Text('Record a set'),
          ),
          const SizedBox(height: 10),
          OutlinedButton.icon(
            onPressed: () => _pick(ImageSource.gallery),
            icon: const Icon(Icons.upload),
            label: const Text('Upload from gallery'),
          ),
        ],
      ),
    );
  }

  Widget _player(VideoPlayerController v) {
    final total = v.value.duration;
    final pos = v.value.position;
    return Column(
      children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(AppRadius.lg),
          child: AspectRatio(
            aspectRatio: v.value.aspectRatio,
            child: Stack(
              alignment: Alignment.center,
              children: [
                VideoPlayer(v),
                GestureDetector(
                  onTap: () => v.value.isPlaying ? v.pause() : v.play(),
                  child: AnimatedOpacity(
                    opacity: v.value.isPlaying ? 0 : 1,
                    duration: const Duration(milliseconds: 150),
                    child: Container(
                      width: 64,
                      height: 64,
                      decoration: const BoxDecoration(
                          color: Colors.black54, shape: BoxShape.circle),
                      child: const Icon(Icons.play_arrow, size: 36),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 10),
        // Timeline with finding markers
        LayoutBuilder(builder: (context, box) {
          final w = box.maxWidth;
          return SizedBox(
            height: 28,
            child: Stack(
              alignment: Alignment.centerLeft,
              children: [
                SliderTheme(
                  data: SliderTheme.of(context).copyWith(
                    trackHeight: 4,
                    activeTrackColor: AppColors.accent,
                    inactiveTrackColor: AppColors.surfaceHi,
                    thumbColor: AppColors.accent,
                    overlayShape: SliderComponentShape.noOverlay,
                  ),
                  child: Slider(
                    value: total.inMilliseconds == 0
                        ? 0
                        : pos.inMilliseconds
                            .clamp(0, total.inMilliseconds)
                            .toDouble(),
                    max: total.inMilliseconds == 0
                        ? 1
                        : total.inMilliseconds.toDouble(),
                    onChanged: (x) =>
                        v.seekTo(Duration(milliseconds: x.round())),
                  ),
                ),
                if (_findings != null && total.inMilliseconds > 0)
                  for (final f in _findings!)
                    Positioned(
                      left: 12 +
                          (w - 24) *
                              (f.at.inMilliseconds / total.inMilliseconds) -
                          4,
                      top: 0,
                      child: IgnorePointer(
                        child: Container(
                          width: 8,
                          height: 8,
                          decoration: BoxDecoration(
                            color: severityColor(f.severity),
                            shape: BoxShape.circle,
                          ),
                        ),
                      ),
                    ),
              ],
            ),
          );
        }),
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(fmtDuration(pos),
                style: const TextStyle(color: AppColors.textDim, fontSize: 12)),
            Text(fmtDuration(total),
                style: const TextStyle(color: AppColors.textDim, fontSize: 12)),
          ],
        ),
      ],
    );
  }

  List<Widget> _results(VideoPlayerController v, List<VideoFinding> findings) {
    final good = findings.where((f) => f.severity == Severity.good).length;
    final score = findings.isEmpty ? 0.0 : good / findings.length * 100;
    final summary = analyzer is BackendPoseAnalyzer
        ? (analyzer as BackendPoseAnalyzer).lastSummary
        : null;
    final detectedEx = summary?['exercise']?.toString();
    final recUser = summary?['recognized_user'] as Map?;
    return [
      const SectionTitle('Analysis'),
      if (summary != null)
        Padding(
          padding: const EdgeInsets.only(bottom: 10),
          child: Panel(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Detected: ${(detectedEx ?? _exercise.id).toUpperCase()}  |  Model: ${summary['model_used'] ?? 'AI'}',
                  style: const TextStyle(
                      color: AppColors.accent,
                      fontSize: 13,
                      fontWeight: FontWeight.w700),
                ),
                const SizedBox(height: 4),
                Text(
                  recUser != null
                      ? 'Athlete: ${recUser['name']} (Age ${recUser['age']}, ${recUser['height_cm']} cm, ${recUser['fitness_level']})'
                      : 'Athlete: Unrecognized (enroll Face ID in Profile)',
                  style: const TextStyle(color: AppColors.textDim, fontSize: 13),
                ),
              ],
            ),
          ),
        ),
      Panel(
        child: Row(
          children: [
            ScoreRing(score: score, size: 80),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('${findings.length} reps analyzed',
                      style: const TextStyle(
                          fontSize: 17, fontWeight: FontWeight.w700)),
                  const SizedBox(height: 2),
                  Text('$good clean, ${findings.length - good} need work',
                      style: const TextStyle(color: AppColors.textDim)),
                ],
              ),
            ),
          ],
        ),
      ),
      const SizedBox(height: 12),
      for (final f in findings)
        Padding(
          padding: const EdgeInsets.only(bottom: 10),
          child: Panel(
            onTap: () {
              v.seekTo(f.at);
              v.pause();
            },
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Container(
                  width: 6,
                  height: 44,
                  decoration: BoxDecoration(
                    color: severityColor(f.severity),
                    borderRadius: BorderRadius.circular(3),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Rep ${f.rep}  -  ${f.title}',
                          style: const TextStyle(
                              fontSize: 15, fontWeight: FontWeight.w700)),
                      const SizedBox(height: 3),
                      Text(f.detail,
                          style: const TextStyle(
                              color: AppColors.textDim, fontSize: 13)),
                    ],
                  ),
                ),
                const SizedBox(width: 8),
                Text(fmtDuration(f.at),
                    style: const TextStyle(
                        color: AppColors.accent, fontWeight: FontWeight.w700)),
              ],
            ),
          ),
        ),
      const SizedBox(height: 8),
      OutlinedButton.icon(
        onPressed: () => setState(() {
          _video?.dispose();
          _video = null;
          _findings = null;
        }),
        icon: const Icon(Icons.refresh),
        label: const Text('Analyze another video'),
      ),
    ];
  }
}
