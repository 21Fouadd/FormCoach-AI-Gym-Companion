import 'package:flutter/material.dart';
import '../../services/auth_service.dart';
import '../../theme/app_theme.dart';
import 'auth_widgets.dart';

class ForgotPasswordScreen extends StatefulWidget {
  final String initialEmail;
  const ForgotPasswordScreen({super.key, this.initialEmail = ''});

  @override
  State<ForgotPasswordScreen> createState() => _ForgotPasswordScreenState();
}

class _ForgotPasswordScreenState extends State<ForgotPasswordScreen> {
  final _form = GlobalKey<FormState>();
  late final _email = TextEditingController(text: widget.initialEmail);
  bool _busy = false;
  bool _sent = false;
  String? _error;

  @override
  void dispose() {
    _email.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    setState(() => _error = null);
    if (!_form.currentState!.validate()) return;
    setState(() => _busy = true);
    try {
      await AuthScope.read(context).sendPasswordReset(_email.text);
      if (mounted) setState(() => _sent = true);
    } on AuthException catch (e) {
      if (mounted) setState(() => _error = e.message);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_sent) {
      return AuthScaffold(
        showBack: true,
        title: 'Check your email',
        subtitle:
            'If an account exists for ${_email.text.trim()}, a reset link is on its way.',
        children: [
          const Icon(Icons.mark_email_read_outlined,
              size: 64, color: AppColors.ok),
          const SizedBox(height: AppSpace.lg),
          FilledButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Back to log in'),
          ),
        ],
      );
    }
    return AuthScaffold(
      showBack: true,
      title: 'Reset password',
      subtitle: 'Enter the email you signed up with and we will send a link.',
      children: [
        if (_error != null) AuthErrorBanner(_error!),
        Form(
          key: _form,
          child: TextFormField(
            controller: _email,
            keyboardType: TextInputType.emailAddress,
            autofillHints: const [AutofillHints.email],
            validator: validateEmail,
            onFieldSubmitted: (_) => _submit(),
            decoration: const InputDecoration(
              labelText: 'Email',
              prefixIcon: Icon(Icons.mail_outline),
            ),
          ),
        ),
        const SizedBox(height: AppSpace.lg),
        BusyButton(label: 'Send reset link', busy: _busy, onPressed: _submit),
      ],
    );
  }
}
