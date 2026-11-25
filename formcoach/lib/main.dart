import 'package:flutter/material.dart';
import 'screens/auth/auth_gate.dart';
import 'services/auth_service.dart';
import 'services/backend_config.dart';
import 'theme/app_theme.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await BackendConfig.load();
  runApp(FormCoachApp(auth: LocalAuthService()));
}

class FormCoachApp extends StatelessWidget {
  /// Swap LocalAuthService for a backend implementation here, once.
  final AuthService auth;

  const FormCoachApp({super.key, required this.auth});

  @override
  Widget build(BuildContext context) {
    return AuthScope(
      auth: auth,
      child: MaterialApp(
        title: 'FormCoach',
        debugShowCheckedModeBanner: false,
        theme: AppTheme.dark,
        home: const AuthGate(),
      ),
    );
  }
}
