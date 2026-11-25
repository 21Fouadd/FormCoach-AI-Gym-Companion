import 'package:flutter/material.dart';
import '../../theme/app_theme.dart';

/// Shared layout for login, sign up and password reset.
class AuthScaffold extends StatelessWidget {
  final String title;
  final String subtitle;
  final List<Widget> children;
  final Widget? footer;
  final bool showBack;

  const AuthScaffold({
    super.key,
    required this.title,
    required this.subtitle,
    required this.children,
    this.footer,
    this.showBack = false,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: showBack ? AppBar() : null,
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.symmetric(
                horizontal: AppSpace.lg, vertical: AppSpace.lg),
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 420),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  if (!showBack) ...[
                    const Align(
                        alignment: Alignment.centerLeft, child: BrandMark()),
                    const SizedBox(height: AppSpace.lg),
                  ],
                  Text(title,
                      style: const TextStyle(
                          fontSize: 28,
                          height: 34 / 28,
                          fontWeight: FontWeight.w800)),
                  const SizedBox(height: AppSpace.sm),
                  Text(subtitle,
                      style: const TextStyle(
                          fontSize: 14, color: AppColors.textDim)),
                  const SizedBox(height: AppSpace.lg),
                  ...children,
                  if (footer != null) ...[
                    const SizedBox(height: AppSpace.lg),
                    footer!,
                  ],
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

/// App mark: lime tile with the training icon and the wordmark.
class BrandMark extends StatelessWidget {
  const BrandMark({super.key});

  @override
  Widget build(BuildContext context) {
    return Row(mainAxisSize: MainAxisSize.min, children: [
      Container(
        width: 44,
        height: 44,
        decoration: BoxDecoration(
          color: AppColors.accent,
          borderRadius: BorderRadius.circular(AppRadius.md),
        ),
        child: const Icon(Icons.fitness_center, color: AppColors.onAccent),
      ),
      const SizedBox(width: 12),
      const Text('FormCoach',
          style: TextStyle(fontSize: 20, fontWeight: FontWeight.w800)),
    ]);
  }
}

/// Error banner: color paired with an icon, per the design rules.
class AuthErrorBanner extends StatelessWidget {
  final String message;
  const AuthErrorBanner(this.message, {super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: AppSpace.md),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppColors.bad.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(AppRadius.md),
        border: Border.all(color: AppColors.bad),
      ),
      child: Row(children: [
        const Icon(Icons.error_outline, color: AppColors.bad, size: 20),
        const SizedBox(width: 10),
        Expanded(child: Text(message)),
      ]),
    );
  }
}

/// Primary button that shows a spinner while [busy].
class BusyButton extends StatelessWidget {
  final String label;
  final bool busy;
  final VoidCallback onPressed;

  const BusyButton({
    super.key,
    required this.label,
    required this.busy,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return FilledButton(
      onPressed: busy ? null : onPressed,
      style: FilledButton.styleFrom(
        disabledBackgroundColor: AppColors.accent.withValues(alpha: 0.6),
      ),
      child: busy
          ? const SizedBox(
              width: 22,
              height: 22,
              child: CircularProgressIndicator(
                  strokeWidth: 2.5, color: AppColors.onAccent),
            )
          : Text(label),
    );
  }
}

/// Password field with a show/hide toggle.
class PasswordField extends StatefulWidget {
  final TextEditingController controller;
  final String label;
  final String? Function(String?)? validator;
  final TextInputAction action;
  final VoidCallback? onSubmitted;
  final Iterable<String> autofillHints;

  const PasswordField({
    super.key,
    required this.controller,
    this.label = 'Password',
    this.validator,
    this.action = TextInputAction.done,
    this.onSubmitted,
    this.autofillHints = const [AutofillHints.password],
  });

  @override
  State<PasswordField> createState() => _PasswordFieldState();
}

class _PasswordFieldState extends State<PasswordField> {
  bool _hidden = true;

  @override
  Widget build(BuildContext context) {
    return TextFormField(
      controller: widget.controller,
      obscureText: _hidden,
      validator: widget.validator,
      textInputAction: widget.action,
      autofillHints: widget.autofillHints,
      onFieldSubmitted: (_) => widget.onSubmitted?.call(),
      decoration: InputDecoration(
        labelText: widget.label,
        prefixIcon: const Icon(Icons.lock_outline),
        suffixIcon: IconButton(
          tooltip: _hidden ? 'Show password' : 'Hide password',
          icon: Icon(_hidden
              ? Icons.visibility_outlined
              : Icons.visibility_off_outlined),
          onPressed: () => setState(() => _hidden = !_hidden),
        ),
      ),
    );
  }
}
