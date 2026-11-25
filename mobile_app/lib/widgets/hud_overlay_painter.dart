import 'package:flutter/material.dart';

/// Paints the live person bounding box, floating registered-user HUD card
/// (Name, Age, Height, Fitness Level), and MediaPipe pose skeleton.
class HudOverlayPainter extends CustomPainter {
  final List<double>? personBox; // [x1, y1, x2, y2] normalized 0..1
  final Map<String, dynamic> landmarks;
  final Map<String, dynamic>? user;
  final double recognitionConfidence;
  final String verdict;

  HudOverlayPainter({
    required this.personBox,
    required this.landmarks,
    required this.user,
    required this.recognitionConfidence,
    required this.verdict,
  });

  static const List<List<int>> _links = [
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
    [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28],
  ];

  @override
  void paint(Canvas canvas, Size size) {
    final isBad = verdict.toLowerCase() == 'incorrect';
    final accent = isBad ? const Color(0xFFF87171) : const Color(0xFF34D399);

    // 1. Skeleton lines & joints
    final linePaint = Paint()
      ..color = accent
      ..strokeWidth = 2.6
      ..style = PaintingStyle.stroke;
    final dotPaint = Paint()
      ..color = const Color(0xFF38BDF8)
      ..style = PaintingStyle.fill;

    for (final link in _links) {
      final a = landmarks[link[0].toString()];
      final b = landmarks[link[1].toString()];
      if (a is List && b is List && a.length >= 2 && b.length >= 2) {
        canvas.drawLine(
          Offset((a[0] as num).toDouble() * size.width, (a[1] as num).toDouble() * size.height),
          Offset((b[0] as num).toDouble() * size.width, (b[1] as num).toDouble() * size.height),
          linePaint,
        );
      }
    }
    landmarks.forEach((_, v) {
      if (v is List && v.length >= 2) {
        canvas.drawCircle(
          Offset((v[0] as num).toDouble() * size.width, (v[1] as num).toDouble() * size.height),
          4.2,
          dotPaint,
        );
      }
    });

    // 2. Person bounding box + Floating Registered User HUD Card
    if (personBox != null && personBox!.length == 4) {
      final rect = Rect.fromLTRB(
        personBox![0] * size.width,
        personBox![1] * size.height,
        personBox![2] * size.width,
        personBox![3] * size.height,
      );
      canvas.drawRect(rect, linePaint);

      final lines = user != null
          ? [
              '${user!['name']} (${(recognitionConfidence * 100).round()}% ID)',
              'Age: ${user!['age']}  |  Height: ${(user!['height_cm'] as num).round()} cm',
              'Fitness Level: ${user!['fitness_level']}',
            ]
          : ['Unregistered Athlete', 'Look at camera for Face ID'];

      const cardWidth = 230.0;
      final cardHeight = lines.length * 18.0 + 16.0;
      final left = rect.left.clamp(8.0, (size.width - cardWidth - 8.0).clamp(8.0, size.width));
      final top = (rect.top - cardHeight - 6.0) >= 8.0
          ? rect.top - cardHeight - 6.0
          : (rect.top + 8.0).clamp(8.0, size.height - cardHeight - 8.0);

      final cardRect = RRect.fromRectAndRadius(
        Rect.fromLTWH(left, top, cardWidth, cardHeight),
        const Radius.circular(8),
      );
      canvas.drawRRect(cardRect, Paint()..color = const Color(0xD90E1420));
      canvas.drawRRect(
        cardRect,
        Paint()
          ..color = accent
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.2,
      );

      for (int i = 0; i < lines.length; i++) {
        final tp = TextPainter(
          text: TextSpan(
            text: lines[i],
            style: TextStyle(
              color: i == 0 ? Colors.white : const Color(0xFFCBD5E1),
              fontSize: 12,
              fontWeight: i == 0 ? FontWeight.w700 : FontWeight.w500,
            ),
          ),
          textDirection: TextDirection.ltr,
        )..layout(maxWidth: cardWidth - 14);
        tp.paint(canvas, Offset(left + 8, top + 8 + i * 18.0));
      }
    }
  }

  @override
  bool shouldRepaint(covariant HudOverlayPainter oldDelegate) => true;
}
