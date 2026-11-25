import 'dart:async';
import 'dart:math' as math;
import 'dart:ui';
import '../models/models.dart';

/// THE INTEGRATION POINT.
///
/// Implement this interface with your own model (TFLite, MediaPipe,
/// a REST endpoint, etc.). The UI only talks to this class.
abstract class PoseAnalyzer {
  /// Load model weights / open connections.
  Future<void> load();

  /// Live analysis. Emit one [FrameAnalysis] per processed camera frame
  /// (or per rep). The UI renders reps, score, phase, issues and skeleton.
  Stream<FrameAnalysis> analyzeLive(Exercise exercise);

  /// Offline analysis of a recorded or uploaded clip.
  Future<List<VideoFinding>> analyzeVideo(String path, Exercise exercise);

  void dispose();
}

/// Fake implementation so the UI runs end to end. Delete once your model is wired.
class MockPoseAnalyzer implements PoseAnalyzer {
  Timer? _timer;
  final _rand = math.Random();

  @override
  Future<void> load() async {
    await Future<void>.delayed(const Duration(milliseconds: 300));
  }

  @override
  Stream<FrameAnalysis> analyzeLive(Exercise exercise) {
    final controller = StreamController<FrameAnalysis>();
    var reps = 0;
    var tick = 0;

    _timer = Timer.periodic(const Duration(milliseconds: 100), (_) {
      tick++;
      final t = (tick % 40) / 40.0; // one rep per 4 seconds
      final depth = math.sin(t * math.pi); // 0..1..0
      if (tick % 40 == 0) reps++;

      final score = 72 + _rand.nextDouble() * 24;
      final issues = <FormIssue>[
        if (depth > 0.8 && score < 80)
          const FormIssue('Knees caving in', Severity.bad),
        if (depth > 0.5 && score < 88)
          const FormIssue('Chest dropping', Severity.warn),
        if (score >= 88) const FormIssue('Good depth', Severity.good),
      ];

      controller.add(FrameAnalysis(
        reps: reps,
        formScore: score,
        phase: depth < 0.15
            ? 'Top'
            : (t < 0.5 ? 'Down' : 'Up'),
        issues: issues,
        landmarks: _mockSkeleton(depth),
      ));
    });

    controller.onCancel = () => _timer?.cancel();
    return controller.stream;
  }

  List<Offset> _mockSkeleton(double depth) {
    final hipY = 0.55 + depth * 0.12;
    return [
      Offset(0.5, 0.18 + depth * 0.1), // 0 head
      Offset(0.5, 0.26 + depth * 0.1), // 1 neck
      Offset(0.42, 0.28 + depth * 0.1), // 2 L shoulder
      Offset(0.58, 0.28 + depth * 0.1), // 3 R shoulder
      Offset(0.38, 0.38 + depth * 0.08), // 4 L elbow
      Offset(0.62, 0.38 + depth * 0.08), // 5 R elbow
      Offset(0.5, hipY), // 6 hips
      Offset(0.44, hipY), // 7 L hip
      Offset(0.56, hipY), // 8 R hip
      Offset(0.43, 0.85), // 9 L foot
      Offset(0.57, 0.85), // 10 R foot
    ];
  }

  @override
  Future<List<VideoFinding>> analyzeVideo(String path, Exercise exercise) async {
    await Future<void>.delayed(const Duration(seconds: 2));
    return const [
      VideoFinding(
        at: Duration(seconds: 3),
        rep: 1,
        title: 'Good rep',
        detail: 'Depth reached parallel, spine neutral.',
        severity: Severity.good,
      ),
      VideoFinding(
        at: Duration(seconds: 7),
        rep: 2,
        title: 'Knees caving in',
        detail: 'Knees moved inward on the way up. Push them out over the toes.',
        severity: Severity.bad,
      ),
      VideoFinding(
        at: Duration(seconds: 11),
        rep: 3,
        title: 'Chest dropping',
        detail: 'Torso leaned forward past 45 degrees at the bottom.',
        severity: Severity.warn,
      ),
      VideoFinding(
        at: Duration(seconds: 15),
        rep: 4,
        title: 'Good rep',
        detail: 'Smooth tempo and full range.',
        severity: Severity.good,
      ),
    ];
  }

  @override
  void dispose() => _timer?.cancel();
}
