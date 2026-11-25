import 'package:flutter/material.dart';
import '../../services/auth_service.dart';
import '../../theme/app_theme.dart';
import '../shell.dart';
import 'auth_widgets.dart';
import 'login_screen.dart';

/// Shows the app when signed in, the login screen otherwise.
class AuthGate extends StatefulWidget {
  const AuthGate({super.key});

  @override
  State<AuthGate> createState() => _AuthGateState();
}

class _AuthGateState extends State<AuthGate> {
  late final Future<void> _restore = AuthScope.read(context).restoreSession();

  @override
  Widget build(BuildContext context) {
    final user = AuthScope.of(context).currentUser;
    return FutureBuilder<void>(
      future: _restore,
      builder: (context, snap) {
        if (snap.connectionState != ConnectionState.done) {
          return const Scaffold(
            body: Center(child: BrandMark()),
          );
        }
        if (snap.hasError) {
          return Scaffold(
            body: Center(
              child: Padding(
                padding: const EdgeInsets.all(AppSpace.lg),
                child: AuthErrorBanner('Could not load your session: ${snap.error}'),
              ),
            ),
          );
        }
        // Keyed so switching users resets the shell's tab state.
        return user == null
            ? const LoginScreen()
            : Shell(key: ValueKey(user.id));
      },
    );
  }
}
