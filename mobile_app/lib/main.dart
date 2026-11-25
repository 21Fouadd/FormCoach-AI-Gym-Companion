import 'package:flutter/material.dart';
import 'screens/live_trainer_screen.dart';
import 'screens/postset_video_screen.dart';
import 'screens/register_user_screen.dart';
import 'services/api_service.dart';

void main() {
  runApp(const AiGymTrainerApp());
}

class AiGymTrainerApp extends StatelessWidget {
  const AiGymTrainerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AI Gym Trainer',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(useMaterial3: true).copyWith(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF34D399), brightness: Brightness.dark),
      ),
      home: const HomeShell(),
    );
  }
}

class HomeShell extends StatefulWidget {
  const HomeShell({super.key});

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  int _tab = 0;

  final _screens = const [
    LiveTrainerScreen(),
    PostSetVideoScreen(),
    RegisterUserScreen(),
  ];

  void _editServerUrl() {
    final ctrl = TextEditingController(text: ApiService.baseUrl);
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Backend Server URL'),
        content: TextField(
          controller: ctrl,
          decoration: const InputDecoration(hintText: 'http://192.168.1.15:8000'),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
          FilledButton(
            onPressed: () {
              setState(() => ApiService.baseUrl = ctrl.text.trim());
              Navigator.pop(ctx);
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('AI Gym Trainer'),
        actions: [
          IconButton(
            tooltip: 'Configure Backend IP',
            icon: const Icon(Icons.dns_outlined),
            onPressed: _editServerUrl,
          ),
        ],
      ),
      body: _screens[_tab],
      bottomNavigationBar: NavigationBar(
        selectedIndex: _tab,
        onDestinationSelected: (i) => setState(() => _tab = i),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.videocam), label: 'Live Trainer'),
          NavigationDestination(icon: Icon(Icons.analytics), label: 'Post-Set'),
          NavigationDestination(icon: Icon(Icons.face_retouching_natural), label: 'Athletes'),
        ],
      ),
    );
  }
}
