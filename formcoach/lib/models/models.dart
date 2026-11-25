import 'package:flutter/material.dart';

enum Severity { good, warn, bad }

class Exercise {
  final String id;
  final String name;
  final String muscle;
  final IconData icon;
  final String cue;

  const Exercise({
    required this.id,
    required this.name,
    required this.muscle,
    required this.icon,
    required this.cue,
  });
}

class FormIssue {
  final String label;
  final Severity severity;
  const FormIssue(this.label, this.severity);
}

/// One analysis tick coming from your model.
/// [landmarks] are normalized (0..1) points in the order of [skeletonEdges].
class FrameAnalysis {
  final int reps;
  final double formScore; // 0..100
  final String phase; // e.g. "Down", "Up", "Hold"
  final List<FormIssue> issues;
  final List<Offset> landmarks;
  final String? athleteName;
  final String? athleteMeta;
  final List<double>? personBox;
  final String? detectedExercise;

  const FrameAnalysis({
    required this.reps,
    required this.formScore,
    required this.phase,
    this.issues = const [],
    this.landmarks = const [],
    this.athleteName,
    this.athleteMeta,
    this.personBox,
    this.detectedExercise,
  });

  static const empty = FrameAnalysis(reps: 0, formScore: 0, phase: 'Ready');
}

/// A flagged moment in a recorded video.
class VideoFinding {
  final Duration at;
  final int rep;
  final String title;
  final String detail;
  final Severity severity;

  const VideoFinding({
    required this.at,
    required this.rep,
    required this.title,
    required this.detail,
    required this.severity,
  });
}

class SessionRecord {
  final String exerciseName;
  final DateTime date;
  final int reps;
  final double avgForm;
  final Duration duration;

  const SessionRecord({
    required this.exerciseName,
    required this.date,
    required this.reps,
    required this.avgForm,
    required this.duration,
  });
}

class SessionResult {
  final Exercise exercise;
  final int reps;
  final double avgForm;
  final Duration duration;
  final List<FormIssue> topIssues;
  final String? videoPath;

  const SessionResult({
    required this.exercise,
    required this.reps,
    required this.avgForm,
    required this.duration,
    required this.topIssues,
    this.videoPath,
  });
}

/// Index pairs into FrameAnalysis.landmarks matching MediaPipe's 12 core joints:
/// 0/1 shoulders (11,12), 2/3 elbows (13,14), 4/5 wrists (15,16),
/// 6/7 hips (23,24), 8/9 knees (25,26), 10/11 ankles (27,28).
const List<List<int>> skeletonEdges = [
  [0, 1],
  [0, 2],
  [2, 4],
  [1, 3],
  [3, 5],
  [0, 6],
  [1, 7],
  [6, 7],
  [6, 8],
  [8, 10],
  [7, 9],
  [9, 11],
];
