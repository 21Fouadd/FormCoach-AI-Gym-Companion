import 'package:flutter/material.dart';
import '../../services/auth_service.dart';
import '../../theme/app_theme.dart';
import 'auth_widgets.dart';
import 'forgot_password_screen.dart';
import 'signup_screen.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _form = GlobalKey<FormState>();
  final _email = TextEditingController();
  final _password = TextEditingController();
  bool _busy = false;
  String? _error;

  @override
  void dispose() {
    _email.dispose();
    _password.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    setState(() => _error = null);
    if (!_form.currentState!.validate()) return;
    setState(() => _busy = true);
    try {
      await AuthScope.read(context)
          .signIn(email: _email.text, password: _password.text);
      // AuthGate swaps to the app; nothing else to do here.
    } on AuthException catch (e) {
      if (mounted) setState(() => _error = e.message);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return AuthScaffold(
      title: 'Welcome back',
      subtitle: 'Log in to keep counting reps and checking your form.',
      footer: Wrap(
        alignment: WrapAlignment.center,
        crossAxisAlignment: WrapCrossAlignment.center,
        children: [
          const Text('New to FormCoach?',
              style: TextStyle(color: AppColors.textDim)),
          TextButton(
            onPressed: _busy
                ? null
                : () => Navigator.push(context,
                    MaterialPageRoute(builder: (_) => const SignupScreen())),
            child: const Text('Create an account',
                style: TextStyle(
                    color: AppColors.accent, fontWeight: FontWeight.w700)),
          ),
        ],
      ),
      children: [
        if (_error != null) AuthErrorBanner(_error!),
        Form(
          key: _form,
          child: AutofillGroup(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                TextFormField(
                  controller: _email,
                  keyboardType: TextInputType.emailAddress,
                  textInputAction: TextInputAction.next,
                  autofillHints: const [AutofillHints.email],
                  validator: validateEmail,
                  decoration: const InputDecoration(
                    labelText: 'Email',
                    prefixIcon: Icon(Icons.mail_outline),
                  ),
                ),
                const SizedBox(height: AppSpace.md),
                PasswordField(
                  controller: _password,
                  // Only check presence at login; length rules are for sign up.
                  validator: (v) =>
                      (v == null || v.isEmpty) ? 'Enter your password' : null,
                  onSubmitted: _submit,
                ),
              ],
            ),
          ),
        ),
        Align(
          alignment: Alignment.centerRight,
          child: TextButton(
            onPressed: _busy
                ? null
                : () => Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (_) =>
                            ForgotPasswordScreen(initialEmail: _email.text),
                      ),
                    ),
            child: const Text('Forgot password?',
                style: TextStyle(color: AppColors.textDim)),
          ),
        ),
        const SizedBox(height: AppSpace.sm),
        BusyButton(label: 'Log in', busy: _busy, onPressed: _submit),
      ],
    );
  }
}
