import 'package:flutter/material.dart';
import '../services/backend_pose_analyzer.dart';
import '../services/pose_analyzer.dart';
import 'community_screen.dart';
import 'exercise_picker_screen.dart';
import 'home_screen.dart';
import 'progress_screen.dart';
import 'video_analysis_screen.dart';

/// Connected to the FastAPI AI backend (live WebSocket, Face ID HUD, and voice cues).
final PoseAnalyzer analyzer = BackendPoseAnalyzer();

class Shell extends StatefulWidget {
  const Shell({super.key});

  @override
  State<Shell> createState() => _ShellState();
}

class _ShellState extends State<Shell> {
  int _index = 0;

  void goTo(int i) => setState(() => _index = i);

  @override
  Widget build(BuildContext context) {
    final pages = [
      HomeScreen(onNavigate: goTo),
      const ExercisePickerScreen(),
      const VideoAnalysisScreen(),
      const CommunityScreen(),
      const ProgressScreen(),
    ];
    return Scaffold(
      body: SafeArea(
        bottom: false,
        child: IndexedStack(index: _index, children: pages),
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _index,
        onDestinationSelected: goTo,
        destinations: const [
          NavigationDestination(icon: Icon(Icons.home_outlined), label: 'Home'),
          NavigationDestination(
              icon: Icon(Icons.fitness_center), label: 'Train'),
          NavigationDestination(
              icon: Icon(Icons.video_camera_back_outlined), label: 'Analyze'),
          NavigationDestination(
              icon: Icon(Icons.groups_outlined), label: 'Community'),
          NavigationDestination(
              icon: Icon(Icons.show_chart), label: 'Progress'),
        ],
      ),
    );
  }
}
