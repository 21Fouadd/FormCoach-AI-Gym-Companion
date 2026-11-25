import 'package:flutter/material.dart';
import '../models/models.dart';

const exercises = <Exercise>[
  Exercise(
    id: 'squat',
    name: 'Back Squat',
    muscle: 'Legs',
    icon: Icons.accessibility_new,
    cue: 'Chest up, knees track over toes, hit parallel.',
  ),
  Exercise(
    id: 'auto',
    name: 'Auto-Detect (AI)',
    muscle: 'Full Body',
    icon: Icons.auto_awesome,
    cue: 'Perform any movement - AI detects the exercise automatically.',
  ),
  Exercise(
    id: 'pushup',
    name: 'Push-up',
    muscle: 'Chest',
    icon: Icons.fitness_center,
    cue: 'Straight line from head to heels, elbows at 45 degrees.',
  ),
  Exercise(
    id: 'deadlift',
    name: 'Bent-Over Row / Deadlift',
    muscle: 'Back',
    icon: Icons.sports_gymnastics,
    cue: 'Neutral spine, hinge at hips, pull/drive cleanly.',
  ),
  Exercise(
    id: 'ohp',
    name: 'Shoulder Press / OHP',
    muscle: 'Shoulders',
    icon: Icons.sports_martial_arts,
    cue: 'Ribs down, glutes tight, press straight overhead.',
  ),
  Exercise(
    id: 'lunge',
    name: 'Lunge',
    muscle: 'Legs',
    icon: Icons.directions_walk,
    cue: 'Torso upright, front knee stable, drop back knee.',
  ),
  Exercise(
    id: 'curl',
    name: 'Bicep Curl',
    muscle: 'Arms',
    icon: Icons.sports_handball,
    cue: 'Elbows pinned, no swinging, full range.',
  ),
  Exercise(
    id: 'arm_abduction',
    name: 'Arm Abduction',
    muscle: 'Shoulders',
    icon: Icons.open_with,
    cue: 'Raise arms evenly to shoulder height without leaning.',
  ),
];

final recentSessions = <SessionRecord>[
  SessionRecord(
    exerciseName: 'Back Squat',
    date: DateTime.now().subtract(const Duration(days: 1)),
    reps: 42,
    avgForm: 88,
    duration: const Duration(minutes: 14, seconds: 20),
  ),
  SessionRecord(
    exerciseName: 'Push-up',
    date: DateTime.now().subtract(const Duration(days: 2)),
    reps: 60,
    avgForm: 79,
    duration: const Duration(minutes: 9, seconds: 5),
  ),
  SessionRecord(
    exerciseName: 'Deadlift',
    date: DateTime.now().subtract(const Duration(days: 4)),
    reps: 25,
    avgForm: 72,
    duration: const Duration(minutes: 18, seconds: 40),
  ),
];

/// Weekly average form score, oldest to newest.
const weeklyForm = <double>[68, 72, 70, 78, 81, 79, 86];
const weeklyLabels = <String>['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7'];
