import 'package:flutter/material.dart';
import '../data/mock_data.dart';
import '../models/models.dart';
import '../theme/app_theme.dart';
import '../widgets/common.dart';
import 'live_session_screen.dart';

class ExercisePickerScreen extends StatefulWidget {
  const ExercisePickerScreen({super.key});

  @override
  State<ExercisePickerScreen> createState() => _ExercisePickerScreenState();
}

class _ExercisePickerScreenState extends State<ExercisePickerScreen> {
  String _filter = 'All';

  @override
  Widget build(BuildContext context) {
    final groups = ['All', ...{for (final e in exercises) e.muscle}];
    final list = _filter == 'All'
        ? exercises
        : exercises.where((e) => e.muscle == _filter).toList();

    return ListView(
      padding: const EdgeInsets.fromLTRB(
          AppSpace.md, AppSpace.sm, AppSpace.md, AppSpace.xl),
      children: [
        const Text('Choose an exercise',
            style: TextStyle(fontSize: 28, fontWeight: FontWeight.w800)),
        const SizedBox(height: 4),
        const Text('Prop your phone up so your full body is in frame.',
            style: TextStyle(color: AppColors.textDim)),
        const SizedBox(height: AppSpace.md),
        SizedBox(
          height: 40,
          child: ListView.separated(
            scrollDirection: Axis.horizontal,
            itemCount: groups.length,
            separatorBuilder: (_, __) => const SizedBox(width: 8),
            itemBuilder: (_, i) {
              final selected = groups[i] == _filter;
              return ChoiceChip(
                label: Text(groups[i]),
                selected: selected,
                onSelected: (_) => setState(() => _filter = groups[i]),
                selectedColor: AppColors.accent,
                backgroundColor: AppColors.surface,
                labelStyle: TextStyle(
                  color: selected ? AppColors.onAccent : AppColors.text,
                  fontWeight: FontWeight.w600,
                ),
                side: const BorderSide(color: AppColors.line),
                showCheckmark: false,
              );
            },
          ),
        ),
        const SizedBox(height: AppSpace.md),
        for (final e in list)
          Padding(
            padding: const EdgeInsets.only(bottom: 10),
            child: _ExerciseCard(exercise: e),
          ),
      ],
    );
  }
}

class _ExerciseCard extends StatelessWidget {
  final Exercise exercise;
  const _ExerciseCard({required this.exercise});

  @override
  Widget build(BuildContext context) {
    return Panel(
      onTap: () => Navigator.push(
        context,
        MaterialPageRoute(
          fullscreenDialog: true,
          builder: (_) => LiveSessionScreen(exercise: exercise),
        ),
      ),
      child: Row(
        children: [
          Container(
            width: 56,
            height: 56,
            decoration: BoxDecoration(
              color: AppColors.surfaceHi,
              borderRadius: BorderRadius.circular(AppRadius.md),
            ),
            child: Icon(exercise.icon, color: AppColors.accent, size: 28),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(exercise.name,
                    style: const TextStyle(
                        fontSize: 17, fontWeight: FontWeight.w700)),
                const SizedBox(height: 2),
                Text(exercise.cue,
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(
                        color: AppColors.textDim, fontSize: 13)),
              ],
            ),
          ),
          const Icon(Icons.chevron_right, color: AppColors.textDim),
        ],
      ),
    );
  }
}
