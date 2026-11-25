import 'dart:async';

import 'package:flutter_test/flutter_test.dart';
import 'package:formcoach/data/mock_data.dart';
import 'package:formcoach/models/models.dart';
import 'package:formcoach/screens/live_session_screen.dart';
import 'package:formcoach/services/pose_analyzer.dart';
import 'package:formcoach/theme/app_theme.dart';
import 'package:formcoach/widgets/common.dart';
import 'package:flutter/material.dart';

/// Analyzer whose frames the test pushes by hand.
class FakeAnalyzer implements PoseAnalyzer {
  final frames = StreamController<FrameAnalysis>.broadcast();

  @override
  Future<void> load() async {}

  @override
  Stream<FrameAnalysis> analyzeLive(Exercise exercise) => frames.stream;

  @override
  Future<List<VideoFinding>> analyzeVideo(String path, Exercise e) async => [];

  @override
  void dispose() => frames.close();
}

void main() {
  late FakeAnalyzer fake;

  setUp(() => fake = FakeAnalyzer());
  tearDown(() => fake.dispose());

  Future<void> pumpLive(WidgetTester tester) async {
    tester.view.physicalSize = const Size(1170, 2532); // phone portrait
    tester.view.devicePixelRatio = 3;
    addTearDown(tester.view.reset);
    await tester.pumpWidget(MaterialApp(
      theme: AppTheme.dark,
      home: LiveSessionScreen(exercise: exercises.first, analyzer: fake),
    ));
    await tester.pump();
  }

  Future<void> emit(WidgetTester tester, FrameAnalysis f) async {
    fake.frames.add(f);
    // One pump delivers the stream event, the next renders the setState.
    await tester.pump();
    await tester.pump();
  }

  /// Dispose the screen so its clock timer is cancelled.
  Future<void> tearDownScreen(WidgetTester tester) =>
      tester.pumpWidget(const SizedBox());

  testWidgets('shows the cue and Start before the set begins', (tester) async {
    await pumpLive(tester);
    expect(find.text('Start set'), findsOneWidget);
    expect(find.text(exercises.first.cue), findsOneWidget);
    expect(find.text('REPS'), findsNothing);
    await tearDownScreen(tester);
  });

  testWidgets('rep counter, phase, score and issues follow the analyzer',
      (tester) async {
    await pumpLive(tester);
    await tester.tap(find.text('Start set'));
    await tester.pump();

    expect(find.text('REPS'), findsOneWidget);
    expect(find.text('0'), findsWidgets);

    await emit(
      tester,
      const FrameAnalysis(
        reps: 3,
        formScore: 64,
        phase: 'Down',
        issues: [FormIssue('Knees caving in', Severity.bad)],
      ),
    );
    expect(find.text('3'), findsOneWidget);
    expect(find.text('DOWN'), findsOneWidget);
    expect(find.text('64'), findsOneWidget);
    expect(find.text('Knees caving in'), findsOneWidget);
    expect(tester.widget<ScoreRing>(find.byType(ScoreRing)).score, 64);

    await emit(tester, const FrameAnalysis(reps: 4, formScore: 90, phase: 'Up'));
    expect(find.text('4'), findsOneWidget);
    expect(find.text('UP'), findsOneWidget);
    expect(find.text('Knees caving in'), findsNothing);

    await tearDownScreen(tester);
  });

  testWidgets('frames are ignored while paused', (tester) async {
    await pumpLive(tester);
    await tester.tap(find.text('Start set'));
    await tester.pump();
    await emit(tester, const FrameAnalysis(reps: 2, formScore: 88, phase: 'Up'));

    await tester.tap(find.byIcon(Icons.pause));
    await tester.pump();
    await emit(tester, const FrameAnalysis(reps: 5, formScore: 88, phase: 'Up'));
    expect(find.text('2'), findsOneWidget);
    expect(find.text('5'), findsNothing);

    await tester.tap(find.byIcon(Icons.play_arrow));
    await tester.pump();
    await emit(tester, const FrameAnalysis(reps: 6, formScore: 88, phase: 'Up'));
    expect(find.text('6'), findsOneWidget);

    await tearDownScreen(tester);
  });

  testWidgets('clock advances once per second', (tester) async {
    await pumpLive(tester);
    await tester.tap(find.text('Start set'));
    await tester.pump();
    expect(find.text('00:00'), findsOneWidget);
    await tester.pump(const Duration(seconds: 3));
    expect(find.text('00:03'), findsOneWidget);
    await tearDownScreen(tester);
  });
}
