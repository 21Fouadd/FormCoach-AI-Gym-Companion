import 'dart:convert';
import 'dart:math' as math;

import 'package:crypto/crypto.dart';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// The signed-in member.
class AppUser {
  final String id;
  final String name;
  final String email;
  final DateTime createdAt;

  const AppUser({
    required this.id,
    required this.name,
    required this.email,
    required this.createdAt,
  });

  String get firstName => name.trim().split(' ').first;

  String get initials => name
      .trim()
      .split(RegExp(r'\s+'))
      .where((w) => w.isNotEmpty)
      .map((w) => w[0].toUpperCase())
      .take(2)
      .join();
}

/// A failure the UI can show as-is.
class AuthException implements Exception {
  final String message;
  const AuthException(this.message);
  @override
  String toString() => message;
}

/// THE AUTH INTEGRATION POINT.
///
/// The UI only talks to this class. [LocalAuthService] keeps accounts on the
/// device; replace it with a Supabase (or other) implementation later without
/// touching the screens. Listeners are notified whenever [currentUser] changes.
abstract class AuthService extends ChangeNotifier {
  AppUser? get currentUser;

  /// Load a saved session, if any. Call once at startup.
  Future<void> restoreSession();

  Future<AppUser> signIn({required String email, required String password});

  Future<AppUser> signUp({
    required String name,
    required String email,
    required String password,
  });

  /// Does not reveal whether an account exists for [email].
  Future<void> sendPasswordReset(String email);

  Future<void> signOut();
}

// ---------------- Validation (shared by UI and services) ----------------

final _emailPattern = RegExp(r'^[^@\s]+@[^@\s]+\.[^@\s]+$');
const minPasswordLength = 8;

String? validateEmail(String? v) {
  final s = v?.trim() ?? '';
  if (s.isEmpty) return 'Enter your email';
  if (!_emailPattern.hasMatch(s)) return 'Enter a valid email';
  return null;
}

String? validatePassword(String? v) {
  final s = v ?? '';
  if (s.isEmpty) return 'Enter your password';
  if (s.length < minPasswordLength) {
    return 'Use at least $minPasswordLength characters';
  }
  return null;
}

String? validateName(String? v) {
  final s = v?.trim() ?? '';
  if (s.isEmpty) return 'Enter your name';
  if (s.length < 2) return 'Name is too short';
  return null;
}

// ---------------- Local implementation ----------------

/// Accounts stored on this device with salted SHA-256 password hashes.
/// Good enough for a prototype and offline demos; NOT a substitute for a real
/// backend (no cross-device accounts, no real password reset email).
class LocalAuthService extends AuthService {
  static const _usersKey = 'auth.users.v1';
  static const _sessionKey = 'auth.session.v1';

  /// Seeded on first run so the app can be demoed without signing up.
  /// Documented in README.md.
  static const demoEmail = 'demo@formcoach.app';
  static const demoPassword = 'formcoach123';
  static const demoName = 'Alex Morgan';

  final bool seedDemoAccount;
  final Duration latency;
  final _rand = math.Random.secure();

  LocalAuthService({
    this.seedDemoAccount = true,
    this.latency = const Duration(milliseconds: 400),
  });

  AppUser? _current;

  @override
  AppUser? get currentUser => _current;

  Future<Map<String, dynamic>> _readUsers(SharedPreferences prefs) async {
    final raw = prefs.getString(_usersKey);
    if (raw == null) return {};
    return (jsonDecode(raw) as Map).cast<String, dynamic>();
  }

  Future<void> _writeUsers(
      SharedPreferences prefs, Map<String, dynamic> users) async {
    await prefs.setString(_usersKey, jsonEncode(users));
  }

  String _newSalt() =>
      base64Url.encode(List<int>.generate(16, (_) => _rand.nextInt(256)));

  String _hash(String password, String salt) =>
      sha256.convert(utf8.encode('$salt:$password')).toString();

  AppUser _toUser(String email, Map<String, dynamic> r) => AppUser(
        id: r['id'] as String,
        name: r['name'] as String,
        email: email,
        createdAt: DateTime.parse(r['createdAt'] as String),
      );

  Map<String, dynamic> _record(String name, String password) {
    final salt = _newSalt();
    return {
      'id': 'u_${DateTime.now().microsecondsSinceEpoch}_${_rand.nextInt(1 << 20)}',
      'name': name.trim(),
      'salt': salt,
      'hash': _hash(password, salt),
      'createdAt': DateTime.now().toIso8601String(),
    };
  }

  String _normalize(String email) => email.trim().toLowerCase();

  @override
  Future<void> restoreSession() async {
    final prefs = await SharedPreferences.getInstance();
    final users = await _readUsers(prefs);
    if (seedDemoAccount && users.isEmpty) {
      users[demoEmail] = _record(demoName, demoPassword);
      await _writeUsers(prefs, users);
    }
    final email = prefs.getString(_sessionKey);
    final r = email == null ? null : users[email];
    _current = r == null ? null : _toUser(email!, (r as Map).cast());
    notifyListeners();
  }

  @override
  Future<AppUser> signIn(
      {required String email, required String password}) async {
    await Future<void>.delayed(latency);
    final key = _normalize(email);
    final prefs = await SharedPreferences.getInstance();
    final users = await _readUsers(prefs);
    final r = (users[key] as Map?)?.cast<String, dynamic>();
    // Same message for unknown email and wrong password.
    if (r == null || _hash(password, r['salt'] as String) != r['hash']) {
      throw const AuthException('Email or password is incorrect');
    }
    await prefs.setString(_sessionKey, key);
    _current = _toUser(key, r);
    notifyListeners();
    return _current!;
  }

  @override
  Future<AppUser> signUp({
    required String name,
    required String email,
    required String password,
  }) async {
    final problem =
        validateName(name) ?? validateEmail(email) ?? validatePassword(password);
    if (problem != null) throw AuthException(problem);

    await Future<void>.delayed(latency);
    final key = _normalize(email);
    final prefs = await SharedPreferences.getInstance();
    final users = await _readUsers(prefs);
    if (users.containsKey(key)) {
      throw const AuthException('An account with this email already exists');
    }
    final r = _record(name, password);
    users[key] = r;
    await _writeUsers(prefs, users);
    await prefs.setString(_sessionKey, key);
    _current = _toUser(key, r);
    notifyListeners();
    return _current!;
  }

  @override
  Future<void> sendPasswordReset(String email) async {
    final problem = validateEmail(email);
    if (problem != null) throw AuthException(problem);
    await Future<void>.delayed(latency);
    // A real backend sends an email here. Locally there is nothing to send.
  }

  @override
  Future<void> signOut() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_sessionKey);
    _current = null;
    notifyListeners();
  }
}

/// Makes the [AuthService] available below it and rebuilds dependents when
/// the signed-in user changes.
class AuthScope extends InheritedNotifier<AuthService> {
  const AuthScope({
    super.key,
    required AuthService auth,
    required super.child,
  }) : super(notifier: auth);

  /// Subscribes the caller to user changes.
  static AuthService of(BuildContext context) => context
      .dependOnInheritedWidgetOfExactType<AuthScope>()!
      .notifier!;

  /// Reads without subscribing (for button handlers).
  static AuthService read(BuildContext context) =>
      context.getInheritedWidgetOfExactType<AuthScope>()!.notifier!;
}
