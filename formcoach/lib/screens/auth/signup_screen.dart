import 'package:flutter/material.dart';
import '../../services/auth_service.dart';
import '../../theme/app_theme.dart';
import 'auth_widgets.dart';

class SignupScreen extends StatefulWidget {
  const SignupScreen({super.key});

  @override
  State<SignupScreen> createState() => _SignupScreenState();
}

class _SignupScreenState extends State<SignupScreen> {
  final _form = GlobalKey<FormState>();
  final _name = TextEditingController();
  final _email = TextEditingController();
  final _password = TextEditingController();
  final _confirm = TextEditingController();
  bool _agreed = false;
  bool _busy = false;
  String? _error;

  @override
  void dispose() {
    for (final c in [_name, _email, _password, _confirm]) {
      c.dispose();
    }
    super.dispose();
  }

  Future<void> _submit() async {
    setState(() => _error = null);
    if (!_form.currentState!.validate()) return;
    if (!_agreed) {
      setState(() => _error = 'Accept the terms to create an account');
      return;
    }
    setState(() => _busy = true);
    try {
      await AuthScope.read(context).signUp(
        name: _name.text,
        email: _email.text,
        password: _password.text,
      );
      // Signed in: drop the auth routes so AuthGate shows the app.
      if (mounted) Navigator.of(context).popUntil((r) => r.isFirst);
    } on AuthException catch (e) {
      if (mounted) setState(() => _error = e.message);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return AuthScaffold(
      showBack: true,
      title: 'Create your account',
      subtitle: 'Track every rep, get form feedback, train with friends.',
      footer: Wrap(
        alignment: WrapAlignment.center,
        crossAxisAlignment: WrapCrossAlignment.center,
        children: [
          const Text('Already a member?',
              style: TextStyle(color: AppColors.textDim)),
          TextButton(
            onPressed: _busy ? null : () => Navigator.pop(context),
            child: const Text('Log in',
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
                  controller: _name,
                  textCapitalization: TextCapitalization.words,
                  textInputAction: TextInputAction.next,
                  autofillHints: const [AutofillHints.name],
                  validator: validateName,
                  decoration: const InputDecoration(
                    labelText: 'Full name',
                    prefixIcon: Icon(Icons.person_outline),
                  ),
                ),
                const SizedBox(height: AppSpace.md),
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
                  validator: validatePassword,
                  action: TextInputAction.next,
                  autofillHints: const [AutofillHints.newPassword],
                ),
                const Padding(
                  padding: EdgeInsets.only(top: 6, left: 4),
                  child: Text('At least $minPasswordLength characters',
                      style:
                          TextStyle(fontSize: 12, color: AppColors.textDim)),
                ),
                const SizedBox(height: AppSpace.md),
                PasswordField(
                  controller: _confirm,
                  label: 'Confirm password',
                  autofillHints: const [AutofillHints.newPassword],
                  validator: (v) =>
                      v != _password.text ? 'Passwords do not match' : null,
                  onSubmitted: _submit,
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: AppSpace.sm),
        CheckboxListTile(
          value: _agreed,
          onChanged: _busy ? null : (v) => setState(() => _agreed = v ?? false),
          controlAffinity: ListTileControlAffinity.leading,
          contentPadding: EdgeInsets.zero,
          activeColor: AppColors.accent,
          checkColor: AppColors.onAccent,
          title: const Text('I agree to the Terms and Privacy Policy',
              style: TextStyle(fontSize: 14)),
        ),
        const SizedBox(height: AppSpace.sm),
        BusyButton(label: 'Create account', busy: _busy, onPressed: _submit),
      ],
    );
  }
}
