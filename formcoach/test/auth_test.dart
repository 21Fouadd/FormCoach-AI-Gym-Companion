import 'package:flutter_test/flutter_test.dart';
import 'package:formcoach/main.dart';
import 'package:formcoach/screens/auth/login_screen.dart';
import 'package:formcoach/screens/profile_screen.dart';
import 'package:formcoach/screens/shell.dart';
import 'package:formcoach/services/auth_service.dart';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

LocalAuthService newAuth({bool seed = false}) =>
    LocalAuthService(seedDemoAccount: seed, latency: Duration.zero);

void main() {
  setUp(() => SharedPreferences.setMockInitialValues({}));

  group('validators', () {
    test('email', () {
      expect(validateEmail(''), isNotNull);
      expect(validateEmail('nope'), isNotNull);
      expect(validateEmail('a@b'), isNotNull);
      expect(validateEmail(' sara@gym.com '), isNull);
    });
    test('password needs $minPasswordLength characters', () {
      expect(validatePassword(''), isNotNull);
      expect(validatePassword('1234567'), isNotNull);
      expect(validatePassword('12345678'), isNull);
    });
    test('name', () {
      expect(validateName('  '), isNotNull);
      expect(validateName('A'), isNotNull);
      expect(validateName('Sara Hassan'), isNull);
    });
  });

  group('LocalAuthService', () {
    test('sign up signs in and persists the session', () async {
      final auth = newAuth();
      await auth.restoreSession();
      expect(auth.currentUser, isNull);

      final u = await auth.signUp(
          name: 'Sara Hassan', email: 'Sara@Gym.com', password: 'squat1234');
      expect(u.email, 'sara@gym.com'); // normalized
      expect(u.initials, 'SH');
      expect(auth.currentUser?.id, u.id);

      // A fresh instance (app restart) restores the same user.
      final again = newAuth();
      await again.restoreSession();
      expect(again.currentUser?.id, u.id);
    });

    test('passwords are not stored in plain text', () async {
      final auth = newAuth();
      await auth.signUp(
          name: 'Sara Hassan', email: 'sara@gym.com', password: 'squat1234');
      final prefs = await SharedPreferences.getInstance();
      final stored = prefs.getKeys().map((k) => '${prefs.get(k)}').join();
      expect(stored.contains('squat1234'), isFalse);
    });

    test('duplicate email is rejected', () async {
      final auth = newAuth();
      await auth.signUp(
          name: 'Sara Hassan', email: 'sara@gym.com', password: 'squat1234');
      expect(
        () => auth.signUp(
            name: 'Other', email: 'SARA@gym.com', password: 'another123'),
        throwsA(isA<AuthException>()),
      );
    });

    test('sign in checks the password, sign out clears the session',
        () async {
      final auth = newAuth();
      await auth.signUp(
          name: 'Sara Hassan', email: 'sara@gym.com', password: 'squat1234');
      await auth.signOut();
      expect(auth.currentUser, isNull);

      await expectLater(
        auth.signIn(email: 'sara@gym.com', password: 'wrongpass'),
        throwsA(isA<AuthException>()),
      );
      await expectLater(
        auth.signIn(email: 'nobody@gym.com', password: 'squat1234'),
        throwsA(isA<AuthException>()),
      );
      final u = await auth.signIn(email: 'sara@gym.com', password: 'squat1234');
      expect(u.name, 'Sara Hassan');

      final restarted = newAuth();
      await auth.signOut();
      await restarted.restoreSession();
      expect(restarted.currentUser, isNull);
    });

    test('demo account is seeded on first run', () async {
      final auth = newAuth(seed: true);
      await auth.restoreSession();
      final u = await auth.signIn(
          email: LocalAuthService.demoEmail,
          password: LocalAuthService.demoPassword);
      expect(u.name, LocalAuthService.demoName);
    });
  });

  group('auth flow', () {
    Future<void> pumpApp(WidgetTester tester, AuthService auth) async {
      tester.view.physicalSize = const Size(1170, 2532);
      tester.view.devicePixelRatio = 3;
      addTearDown(tester.view.reset);
      await tester.pumpWidget(FormCoachApp(auth: auth));
      await tester.pumpAndSettle();
    }

    Future<void> fill(WidgetTester tester, String label, String text) =>
        tester.enterText(find.widgetWithText(TextFormField, label), text);

    testWidgets('signed out users land on login and see validation',
        (tester) async {
      await pumpApp(tester, newAuth());
      expect(find.byType(LoginScreen), findsOneWidget);

      await tester.tap(find.text('Log in'));
      await tester.pump();
      expect(find.text('Enter your email'), findsOneWidget);
      expect(find.text('Enter your password'), findsOneWidget);
    });

    testWidgets('wrong password shows an error banner', (tester) async {
      await pumpApp(tester, newAuth(seed: true));
      await fill(tester, 'Email', LocalAuthService.demoEmail);
      await fill(tester, 'Password', 'not-the-password');
      await tester.tap(find.text('Log in'));
      await tester.pumpAndSettle();
      expect(find.text('Email or password is incorrect'), findsOneWidget);
      expect(find.byType(Shell), findsNothing);
    });

    testWidgets('sign up, sign out, log back in', (tester) async {
      await pumpApp(tester, newAuth());

      await tester.tap(find.text('Create an account'));
      await tester.pumpAndSettle();
      await fill(tester, 'Full name', 'Omar Nabil');
      await fill(tester, 'Email', 'omar@gym.com');
      await fill(tester, 'Password', 'deadlift99');
      await fill(tester, 'Confirm password', 'deadlift98');
      await tester.tap(find.text('Create account'));
      await tester.pump();
      expect(find.text('Passwords do not match'), findsOneWidget);

      await fill(tester, 'Confirm password', 'deadlift99');
      await tester.tap(find.text('Create account'));
      await tester.pump();
      expect(find.text('Accept the terms to create an account'),
          findsOneWidget);

      await tester.tap(find.byType(Checkbox));
      await tester.tap(find.text('Create account'));
      await tester.pumpAndSettle();
      expect(find.byType(Shell), findsOneWidget);
      expect(find.text('Omar'), findsOneWidget); // home greeting

      // Profile -> Sign out -> confirm.
      await tester.tap(find.byIcon(Icons.person));
      await tester.pumpAndSettle();
      expect(find.text('omar@gym.com'), findsOneWidget);
      await tester.scrollUntilVisible(find.text('Sign out'), 200,
          scrollable: find.descendant(
              of: find.byType(ProfileScreen),
              matching: find.byType(Scrollable)));
      await tester.ensureVisible(find.text('Sign out'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Sign out'));
      await tester.pumpAndSettle();
      await tester.tap(find.widgetWithText(TextButton, 'Sign out'));
      await tester.pumpAndSettle();
      expect(find.byType(LoginScreen), findsOneWidget);

      await fill(tester, 'Email', 'omar@gym.com');
      await fill(tester, 'Password', 'deadlift99');
      await tester.tap(find.text('Log in'));
      await tester.pumpAndSettle();
      expect(find.byType(Shell), findsOneWidget);
    });

    testWidgets('forgot password confirms without revealing accounts',
        (tester) async {
      await pumpApp(tester, newAuth());
      await tester.tap(find.text('Forgot password?'));
      await tester.pumpAndSettle();
      await fill(tester, 'Email', 'someone@gym.com');
      await tester.tap(find.text('Send reset link'));
      await tester.pumpAndSettle();
      expect(find.text('Check your email'), findsOneWidget);
      await tester.tap(find.text('Back to log in'));
      await tester.pumpAndSettle();
      expect(find.byType(LoginScreen), findsOneWidget);
    });
  });
}
