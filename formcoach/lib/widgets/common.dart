import 'dart:math' as math;
import 'package:flutter/material.dart';
import '../models/models.dart';
import '../theme/app_theme.dart';

Color severityColor(Severity s) => switch (s) {
      Severity.good => AppColors.ok,
      Severity.warn => AppColors.warn,
      Severity.bad => AppColors.bad,
    };

Color scoreColor(double score) => score >= 85
    ? AppColors.ok
    : score >= 70
        ? AppColors.warn
        : AppColors.bad;

class Panel extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry padding;
  final VoidCallback? onTap;
  final Color? color;

  const Panel({
    super.key,
    required this.child,
    this.padding = const EdgeInsets.all(AppSpace.md),
    this.onTap,
    this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Material(
      color: color ?? AppColors.surface,
      borderRadius: BorderRadius.circular(AppRadius.lg),
      child: InkWell(
        borderRadius: BorderRadius.circular(AppRadius.lg),
        onTap: onTap,
        child: Padding(padding: padding, child: child),
      ),
    );
  }
}

class SectionTitle extends StatelessWidget {
  final String title;
  final String? action;
  final VoidCallback? onAction;

  const SectionTitle(this.title, {super.key, this.action, this.onAction});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(top: AppSpace.lg, bottom: AppSpace.sm),
      child: Row(
        children: [
          Expanded(
            child: Text(
              title,
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w700),
            ),
          ),
          if (action != null)
            TextButton(
              onPressed: onAction,
              child: Text(action!,
                  style: const TextStyle(color: AppColors.accent)),
            ),
        ],
      ),
    );
  }
}

class StatTile extends StatelessWidget {
  final String value;
  final String label;
  final IconData icon;

  const StatTile({
    super.key,
    required this.value,
    required this.label,
    required this.icon,
  });

  @override
  Widget build(BuildContext context) {
    return Panel(
      padding: const EdgeInsets.all(14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 20, color: AppColors.accent),
          const SizedBox(height: 10),
          Text(value,
              style: const TextStyle(fontSize: 24, fontWeight: FontWeight.w800)),
          const SizedBox(height: 2),
          Text(label,
              style: const TextStyle(fontSize: 12, color: AppColors.textDim)),
        ],
      ),
    );
  }
}

class SeverityChip extends StatelessWidget {
  final FormIssue issue;
  const SeverityChip(this.issue, {super.key});

  @override
  Widget build(BuildContext context) {
    final c = severityColor(issue.severity);
    final icon = switch (issue.severity) {
      Severity.good => Icons.check_circle,
      Severity.warn => Icons.warning_amber_rounded,
      Severity.bad => Icons.error,
    };
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.55),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: c, width: 1.5),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: c),
          const SizedBox(width: 6),
          Text(issue.label,
              style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
        ],
      ),
    );
  }
}

class ScoreRing extends StatelessWidget {
  final double score; // 0..100
  final double size;
  final double stroke;
  final bool showLabel;

  const ScoreRing({
    super.key,
    required this.score,
    this.size = 72,
    this.stroke = 8,
    this.showLabel = true,
  });

  @override
  Widget build(BuildContext context) {
    final color = scoreColor(score);
    return SizedBox(
      width: size,
      height: size,
      child: CustomPaint(
        painter: _RingPainter(score / 100, color, stroke),
        child: Center(
          child: showLabel
              ? Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      score.round().toString(),
                      style: TextStyle(
                        fontSize: size * 0.32,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    Text('FORM',
                        style: TextStyle(
                          fontSize: size * 0.12,
                          letterSpacing: 1,
                          color: AppColors.textDim,
                        )),
                  ],
                )
              : null,
        ),
      ),
    );
  }
}

class _RingPainter extends CustomPainter {
  final double progress;
  final Color color;
  final double stroke;
  _RingPainter(this.progress, this.color, this.stroke);

  @override
  void paint(Canvas canvas, Size size) {
    final rect = Offset.zero & size;
    final inset = rect.deflate(stroke / 2);
    final bg = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = stroke
      ..color = Colors.white.withValues(alpha: 0.12);
    final fg = Paint()
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeWidth = stroke
      ..color = color;
    canvas.drawArc(inset, 0, math.pi * 2, false, bg);
    canvas.drawArc(inset, -math.pi / 2, math.pi * 2 * progress.clamp(0, 1),
        false, fg);
  }

  @override
  bool shouldRepaint(_RingPainter old) =>
      old.progress != progress || old.color != color;
}

class MiniBarChart extends StatelessWidget {
  final List<double> values;
  final List<String> labels;
  final double height;

  const MiniBarChart({
    super.key,
    required this.values,
    required this.labels,
    this.height = 120,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: height + 22,
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          for (var i = 0; i < values.length; i++)
            Expanded(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 4),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.end,
                  children: [
                    Container(
                      height: height * (values[i] / 100),
                      decoration: BoxDecoration(
                        color: i == values.length - 1
                            ? AppColors.accent
                            : AppColors.surfaceHi,
                        borderRadius: BorderRadius.circular(6),
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(labels[i],
                        style: const TextStyle(
                            fontSize: 11, color: AppColors.textDim)),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}

class SkeletonPainter extends CustomPainter {
  final List<Offset> landmarks;
  final Color color;
  SkeletonPainter(this.landmarks, {this.color = AppColors.accent});

  @override
  void paint(Canvas canvas, Size size) {
    if (landmarks.isEmpty) return;
    final line = Paint()
      ..color = color.withValues(alpha: 0.9)
      ..strokeWidth = 3
      ..strokeCap = StrokeCap.round;
    final dot = Paint()..color = Colors.white;
    Offset p(int i) =>
        Offset(landmarks[i].dx * size.width, landmarks[i].dy * size.height);

    bool valid(int i) =>
        i < landmarks.length && landmarks[i].dx >= 0 && landmarks[i].dy >= 0;

    for (final e in skeletonEdges) {
      if (valid(e[0]) && valid(e[1])) {
        canvas.drawLine(p(e[0]), p(e[1]), line);
      }
    }
    for (var i = 0; i < landmarks.length; i++) {
      if (valid(i)) {
        canvas.drawCircle(p(i), 5, dot);
      }
    }
  }

  @override
  bool shouldRepaint(SkeletonPainter old) => old.landmarks != landmarks;
}

String fmtDuration(Duration d) {
  final m = d.inMinutes.remainder(60).toString().padLeft(2, '0');
  final s = d.inSeconds.remainder(60).toString().padLeft(2, '0');
  return '$m:$s';
}

String fmtDay(DateTime d) {
  final diff = DateTime.now().difference(d).inDays;
  if (diff <= 0) return 'Today';
  if (diff == 1) return 'Yesterday';
  return '$diff days ago';
}
