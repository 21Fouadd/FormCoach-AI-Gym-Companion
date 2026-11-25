import 'package:flutter_test/flutter_test.dart';
import 'package:formcoach/models/models.dart';
import 'package:formcoach/theme/app_theme.dart';
import 'package:formcoach/widgets/common.dart';
import 'package:flutter/material.dart';

void main() {
  group('scoreColor: >= 85 ok, 70-84 warn, < 70 bad', () {
    test('ok band', () {
      expect(scoreColor(100), AppColors.ok);
      expect(scoreColor(85), AppColors.ok);
    });

    test('warn band', () {
      expect(scoreColor(84.9), AppColors.warn);
      expect(scoreColor(70), AppColors.warn);
    });

    test('bad band', () {
      expect(scoreColor(69.9), AppColors.bad);
      expect(scoreColor(0), AppColors.bad);
    });
  });

  test('severityColor maps each severity to its status color', () {
    expect(severityColor(Severity.good), AppColors.ok);
    expect(severityColor(Severity.warn), AppColors.warn);
    expect(severityColor(Severity.bad), AppColors.bad);
  });

  testWidgets('ScoreRing pairs the color with a numeric label',
      (tester) async {
    await tester.pumpWidget(const MaterialApp(
      home: Scaffold(body: Center(child: ScoreRing(score: 84.6))),
    ));
    expect(find.text('85'), findsOneWidget);
    expect(find.text('FORM'), findsOneWidget);
  });

  testWidgets('SeverityChip pairs the color with an icon', (tester) async {
    await tester.pumpWidget(const MaterialApp(
      home: Scaffold(
        body: Column(children: [
          SeverityChip(FormIssue('Good depth', Severity.good)),
          SeverityChip(FormIssue('Chest dropping', Severity.warn)),
          SeverityChip(FormIssue('Knees caving in', Severity.bad)),
        ]),
      ),
    ));
    Icon iconFor(IconData data) => tester.widget<Icon>(find.byIcon(data));
    expect(iconFor(Icons.check_circle).color, AppColors.ok);
    expect(iconFor(Icons.warning_amber_rounded).color, AppColors.warn);
    expect(iconFor(Icons.error).color, AppColors.bad);
  });
}
