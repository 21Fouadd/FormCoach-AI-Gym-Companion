import 'package:flutter/material.dart';
import '../data/mock_data.dart';
import '../theme/app_theme.dart';
import '../widgets/common.dart';

class ProgressScreen extends StatelessWidget {
  const ProgressScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.fromLTRB(
          AppSpace.md, AppSpace.sm, AppSpace.md, AppSpace.xl),
      children: [
        const Text('Progress',
            style: TextStyle(fontSize: 28, fontWeight: FontWeight.w800)),
        const SizedBox(height: AppSpace.md),
        Panel(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('Form score trend',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700)),
              const SizedBox(height: 2),
              const Text('Weekly average, last 7 weeks',
                  style: TextStyle(color: AppColors.textDim, fontSize: 13)),
              const SizedBox(height: AppSpace.md),
              MiniBarChart(values: weeklyForm, labels: weeklyLabels),
            ],
          ),
        ),
        const SectionTitle('Most common issues'),
        const _IssueRow(label: 'Knees caving in', count: 14, fraction: 1.0),
        const _IssueRow(label: 'Chest dropping', count: 9, fraction: 0.64),
        const _IssueRow(label: 'Rounded lower back', count: 5, fraction: 0.36),
        const SectionTitle('History'),
        for (final s in recentSessions)
          Padding(
            padding: const EdgeInsets.only(bottom: 10),
            child: Panel(
              child: Row(
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(s.exerciseName,
                            style: const TextStyle(
                                fontSize: 16, fontWeight: FontWeight.w700)),
                        const SizedBox(height: 2),
                        Text('${fmtDay(s.date)}  -  ${s.reps} reps',
                            style: const TextStyle(
                                color: AppColors.textDim, fontSize: 13)),
                      ],
                    ),
                  ),
                  Text('${s.avgForm.round()}',
                      style: TextStyle(
                          fontSize: 22,
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

class _IssueRow extends StatelessWidget {
  final String label;
  final int count;
  final double fraction;
  const _IssueRow(
      {required this.label, required this.count, required this.fraction});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Panel(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                    child: Text(label,
                        style: const TextStyle(
                            fontSize: 15, fontWeight: FontWeight.w600))),
                Text('$count times',
                    style: const TextStyle(
                        color: AppColors.textDim, fontSize: 13)),
              ],
            ),
            const SizedBox(height: 10),
            ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: fraction,
                minHeight: 6,
                color: AppColors.warn,
                backgroundColor: AppColors.surfaceHi,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
