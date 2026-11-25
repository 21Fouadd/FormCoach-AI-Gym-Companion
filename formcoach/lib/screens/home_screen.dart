import 'package:flutter/material.dart';
import '../services/auth_service.dart';
import '../data/mock_data.dart';
import '../theme/app_theme.dart';
import '../widgets/common.dart';
import 'profile_screen.dart';

class HomeScreen extends StatelessWidget {
  /// Switches the bottom tab (1 = Train, 2 = Analyze, 3 = Progress).
  final ValueChanged<int> onNavigate;
  const HomeScreen({super.key, required this.onNavigate});

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.fromLTRB(
          AppSpace.md, AppSpace.sm, AppSpace.md, AppSpace.xl),
      children: [
        Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(_greeting(DateTime.now().hour),
                      style: const TextStyle(
                          color: AppColors.textDim, fontSize: 14)),
                  const SizedBox(height: 2),
                  Text(AuthScope.of(context).currentUser?.firstName ?? 'there',
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                          fontSize: 28, fontWeight: FontWeight.w800)),
                ],
              ),
            ),
            GestureDetector(
              onTap: () => Navigator.push(
                context,
                MaterialPageRoute(builder: (_) => const ProfileScreen()),
              ),
              child: const CircleAvatar(
                radius: 22,
                backgroundColor: AppColors.surfaceHi,
                child: Icon(Icons.person, color: AppColors.text),
              ),
            ),
          ],
        ),
        const SizedBox(height: AppSpace.lg),

        // Today's plan
        Panel(
          color: AppColors.accent,
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text("TODAY'S PLAN",
                  style: TextStyle(
                      color: AppColors.onAccent,
                      fontSize: 12,
                      fontWeight: FontWeight.w800,
                      letterSpacing: 1.2)),
              const SizedBox(height: 8),
              const Text('Leg Day',
                  style: TextStyle(
                      color: AppColors.onAccent,
                      fontSize: 30,
                      fontWeight: FontWeight.w900)),
              const SizedBox(height: 4),
              const Text('Back Squat, Lunge, Deadlift  -  45 min',
                  style: TextStyle(color: AppColors.onAccent)),
              const SizedBox(height: 18),
              FilledButton(
                style: FilledButton.styleFrom(
                  backgroundColor: AppColors.onAccent,
                  foregroundColor: AppColors.accent,
                ),
                onPressed: () => onNavigate(1),
                child: const Text('Start session'),
              ),
            ],
          ),
        ),

        const SectionTitle('This week'),
        const Row(
          children: [
            Expanded(
                child: StatTile(
                    value: '127',
                    label: 'Reps counted',
                    icon: Icons.repeat)),
            SizedBox(width: 12),
            Expanded(
                child: StatTile(
                    value: '86',
                    label: 'Avg form score',
                    icon: Icons.verified_outlined)),
            SizedBox(width: 12),
            Expanded(
                child: StatTile(
                    value: '5',
                    label: 'Day streak',
                    icon: Icons.local_fire_department_outlined)),
          ],
        ),

        const SectionTitle('Quick actions'),
        Row(
          children: [
            Expanded(
              child: _QuickAction(
                icon: Icons.videocam_outlined,
                title: 'Live form check',
                subtitle: 'Camera + rep counter',
                onTap: () => onNavigate(1),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _QuickAction(
                icon: Icons.upload_file_outlined,
                title: 'Analyze a video',
                subtitle: 'Upload your set',
                onTap: () => onNavigate(2),
              ),
            ),
          ],
        ),

        SectionTitle('Recent sessions',
            action: 'See all', onAction: () => onNavigate(4)),
        for (final s in recentSessions)
          Padding(
            padding: const EdgeInsets.only(bottom: 10),
            child: Panel(
              child: Row(
                children: [
                  ScoreRing(score: s.avgForm, size: 52, stroke: 6, showLabel: false),
                  const SizedBox(width: 14),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(s.exerciseName,
                            style: const TextStyle(
                                fontSize: 16, fontWeight: FontWeight.w700)),
                        const SizedBox(height: 2),
                        Text(
                            '${fmtDay(s.date)}  -  ${s.reps} reps  -  ${fmtDuration(s.duration)}',
                            style: const TextStyle(
                                color: AppColors.textDim, fontSize: 13)),
                      ],
                    ),
                  ),
                  Text('${s.avgForm.round()}',
                      style: TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.w800,
                          color: scoreColor(s.avgForm))),
                ],
              ),
            ),
          ),
      ],
    );
  }
}

class _QuickAction extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final VoidCallback onTap;

  const _QuickAction({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Panel(
      onTap: onTap,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: AppColors.accent, size: 28),
          const SizedBox(height: 14),
          Text(title,
              style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w700)),
          const SizedBox(height: 2),
          Text(subtitle,
              style: const TextStyle(fontSize: 12, color: AppColors.textDim)),
        ],
      ),
    );
  }
}

String _greeting(int hour) => hour < 12
    ? 'Good morning'
    : hour < 18
        ? 'Good afternoon'
        : 'Good evening';
