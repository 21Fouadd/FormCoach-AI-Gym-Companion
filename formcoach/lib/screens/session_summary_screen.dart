import 'package:flutter/material.dart';
import '../models/models.dart';
import '../theme/app_theme.dart';
import '../widgets/common.dart';
import 'video_analysis_screen.dart';

class SessionSummaryScreen extends StatelessWidget {
  final SessionResult result;
  const SessionSummaryScreen({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        automaticallyImplyLeading: false,
        title: const Text('Set complete',
            style: TextStyle(fontWeight: FontWeight.w800)),
      ),
      body: ListView(
        padding: const EdgeInsets.all(AppSpace.md),
        children: [
          Panel(
            padding: const EdgeInsets.all(24),
            child: Column(
              children: [
                Text(result.exercise.name,
                    style: const TextStyle(
                        color: AppColors.textDim, fontSize: 15)),
                const SizedBox(height: 16),
                ScoreRing(score: result.avgForm, size: 140, stroke: 12),
                const SizedBox(height: 20),
                Row(
                  children: [
                    Expanded(
                        child: _Metric(value: '${result.reps}', label: 'Reps')),
                    Expanded(
                        child: _Metric(
                            value: fmtDuration(result.duration),
                            label: 'Time')),
                  ],
                ),
              ],
            ),
          ),
          const SectionTitle('Fix these next time'),
          if (result.topIssues.isEmpty)
            const Panel(child: Text('No form issues detected. Nice work.'))
          else
            for (final i in result.topIssues)
              Padding(
                padding: const EdgeInsets.only(bottom: 10),
                child: Panel(
                  child: Row(
                    children: [
                      Icon(Icons.circle, size: 12, color: severityColor(i.severity)),
                      const SizedBox(width: 12),
                      Expanded(
                          child: Text(i.label,
                              style: const TextStyle(
                                  fontSize: 15, fontWeight: FontWeight.w600))),
                    ],
                  ),
                ),
              ),
          const SizedBox(height: AppSpace.lg),
          if (result.videoPath != null) ...[
            OutlinedButton.icon(
              icon: const Icon(Icons.slow_motion_video),
              label: const Text('Review recorded video'),
              onPressed: () => Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (_) => Scaffold(
                    appBar: AppBar(title: const Text('Video review')),
                    body: VideoAnalysisScreen(
                      initialPath: result.videoPath,
                      initialExercise: result.exercise,
                    ),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 12),
          ],
          FilledButton(
            onPressed: () => Navigator.popUntil(context, (r) => r.isFirst),
            child: const Text('Save and finish'),
          ),
        ],
      ),
    );
  }
}

class _Metric extends StatelessWidget {
  final String value;
  final String label;
  const _Metric({required this.value, required this.label});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text(value,
            style: const TextStyle(fontSize: 30, fontWeight: FontWeight.w800)),
        Text(label,
            style: const TextStyle(color: AppColors.textDim, fontSize: 13)),
      ],
    );
  }
}
