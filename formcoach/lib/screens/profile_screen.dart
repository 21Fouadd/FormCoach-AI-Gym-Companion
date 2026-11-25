import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/auth_service.dart';
import '../services/backend_config.dart';
import '../theme/app_theme.dart';
import '../widgets/common.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  bool _voice = BackendConfig.voiceEnabled;
  bool _haptics = true;
  bool _skeleton = true;
  bool _autoRecord = false;

  late final TextEditingController _serverCtrl =
      TextEditingController(text: BackendConfig.baseUrl);
  final _ageCtrl = TextEditingController(text: '22');
  final _heightCtrl = TextEditingController(text: '178');
  final _weightCtrl = TextEditingController(text: '75');
  String _serverStatus = 'Tap Test Connection to check the AI backend.';
  String _faceStatus = 'Not enrolled yet — capture a front face photo.';
  bool _busy = false;

  static const _months = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
  ];

  @override
  void dispose() {
    _serverCtrl.dispose();
    _ageCtrl.dispose();
    _heightCtrl.dispose();
    _weightCtrl.dispose();
    super.dispose();
  }

  Future<void> _saveAndTestServer([String? overrideUrl]) async {
    if (overrideUrl != null) {
      _serverCtrl.text = overrideUrl;
    }
    setState(() {
      _busy = true;
      _serverStatus = 'Connecting to ${_serverCtrl.text.trim()}...';
    });
    await BackendConfig.setBaseUrl(_serverCtrl.text);
    try {
      final health = await BackendConfig.checkHealth();
      final models = health['loaded_models'] as Map? ?? {};
      final faceOk = (health['face_recognition'] as Map?)?['ready'] == true;
      if (mounted) {
        setState(() {
          _serverStatus =
              'Connected! Squat: ${models['squat'] ?? 'OK'} | Push: ${models['push'] ?? 'OK'} | Face ID: ${faceOk ? 'Ready' : 'Loading'}';
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() => _serverStatus = 'Connection failed: $e');
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _enrollFace(ImageSource source, String defaultName) async {
    final picked = await ImagePicker()
        .pickImage(source: source, maxWidth: 800, imageQuality: 85);
    if (picked == null) return;

    setState(() {
      _busy = true;
      _faceStatus = 'Extracting SFace embedding & predicting Fitness Level...';
    });
    try {
      final res = await BackendConfig.registerFaceProfile(
        name: defaultName,
        age: int.tryParse(_ageCtrl.text) ?? 22,
        heightCm: double.tryParse(_heightCtrl.text) ?? 178,
        weightKg: double.tryParse(_weightCtrl.text) ?? 75,
        fitnessLevel: 'Auto',
        photoPath: picked.path,
      );
      final u = res['user'] as Map? ?? {};
      final pred = res['fitness_prediction'] as Map? ?? {};
      if (mounted) {
        setState(() {
          _faceStatus =
              'Enrolled ${u['name']}! AI Fitness Level: ${u['fitness_level']} (${pred['predicted_by'] ?? 'SmartFit'})';
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() => _faceStatus = 'Face enrollment failed: $e');
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _confirmSignOut() async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: AppColors.surface,
        title: const Text('Sign out?'),
        content: const Text('You will need to log in again to use FormCoach.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel',
                style: TextStyle(color: AppColors.textDim)),
          ),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Sign out',
                style: TextStyle(color: AppColors.bad)),
          ),
        ],
      ),
    );
    if (ok != true || !mounted) return;
    final auth = AuthScope.read(context);
    Navigator.of(context).popUntil((r) => r.isFirst);
    await auth.signOut();
  }

  @override
  Widget build(BuildContext context) {
    final user = AuthScope.of(context).currentUser;
    return Scaffold(
      appBar: AppBar(title: const Text('Profile')),
      body: ListView(
        padding: const EdgeInsets.all(AppSpace.md),
        children: [
          Center(
            child: CircleAvatar(
              radius: 44,
              backgroundColor: AppColors.surfaceHi,
              foregroundColor: AppColors.text,
              child: user == null
                  ? const Icon(Icons.person, size: 44)
                  : Text(user.initials,
                      style: const TextStyle(
                          fontSize: 30, fontWeight: FontWeight.w800)),
            ),
          ),
          const SizedBox(height: 12),
          Center(
            child: Text(user?.name ?? 'Guest',
                style: const TextStyle(
                    fontSize: 22, fontWeight: FontWeight.w800)),
          ),
          if (user != null) ...[
            Center(
              child: Text(user.email,
                  style: const TextStyle(color: AppColors.textDim)),
            ),
            Center(
              child: Text(
                  'Gym member since ${_months[user.createdAt.month - 1]} ${user.createdAt.year}',
                  style: const TextStyle(color: AppColors.textDim)),
            ),
          ],

          const SectionTitle('Backend Server & Face ID'),
          Panel(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                TextField(
                  controller: _serverCtrl,
                  decoration: const InputDecoration(
                    labelText: 'FastAPI Server URL',
                    hintText: 'http://192.168.1.3:8000',
                  ),
                ),
                const SizedBox(height: 8),
                Wrap(
                  spacing: 8,
                  children: [
                    ActionChip(
                      label: const Text('Wi-Fi IP'),
                      onPressed: _busy
                          ? null
                          : () => _saveAndTestServer('http://192.168.1.3:8000'),
                    ),
                    ActionChip(
                      label: const Text('Android Emulator'),
                      onPressed: _busy
                          ? null
                          : () => _saveAndTestServer('http://10.0.2.2:8000'),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                OutlinedButton.icon(
                  onPressed: _busy ? null : () => _saveAndTestServer(),
                  icon: const Icon(Icons.wifi_tethering),
                  label: const Text('Save & Test Connection'),
                ),
                const SizedBox(height: 8),
                Text(_serverStatus,
                    style:
                        const TextStyle(fontSize: 12, color: AppColors.textDim)),
                const Divider(height: 24),
                Row(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: _ageCtrl,
                        keyboardType: TextInputType.number,
                        decoration: const InputDecoration(labelText: 'Age'),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: TextField(
                        controller: _heightCtrl,
                        keyboardType: TextInputType.number,
                        decoration:
                            const InputDecoration(labelText: 'Height (cm)'),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: TextField(
                        controller: _weightCtrl,
                        keyboardType: TextInputType.number,
                        decoration:
                            const InputDecoration(labelText: 'Weight (kg)'),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 10),
                Row(
                  children: [
                    Expanded(
                      child: FilledButton.icon(
                        onPressed: _busy
                            ? null
                            : () => _enrollFace(
                                ImageSource.camera, user?.name ?? 'Athlete'),
                        icon: const Icon(Icons.face_retouching_natural),
                        label: const Text('Enroll Face Camera'),
                      ),
                    ),
                    const SizedBox(width: 8),
                    IconButton.outlined(
                      tooltip: 'Pick face photo from gallery',
                      onPressed: _busy
                          ? null
                          : () => _enrollFace(
                              ImageSource.gallery, user?.name ?? 'Athlete'),
                      icon: const Icon(Icons.photo_library_outlined),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                Text(_faceStatus,
                    style:
                        const TextStyle(fontSize: 12, color: AppColors.accent)),
              ],
            ),
          ),

          const SectionTitle('Coaching'),
          Panel(
            padding: EdgeInsets.zero,
            child: Column(
              children: [
                _tile('Voice cues', 'Spoken feedback during sets', _voice,
                    (v) {
                  setState(() => _voice = v);
                  BackendConfig.setVoiceEnabled(v);
                }),
                _tile('Haptic alerts', 'Vibrate on bad form', _haptics,
                    (v) => setState(() => _haptics = v)),
                _tile('Show skeleton overlay', 'Draw tracked joints', _skeleton,
                    (v) => setState(() => _skeleton = v)),
                _tile('Auto-record sets', 'Save video for review', _autoRecord,
                    (v) => setState(() => _autoRecord = v)),
              ],
            ),
          ),
          const SectionTitle('Account'),
          Panel(
            padding: EdgeInsets.zero,
            child: Column(
              children: [
                const ListTile(
                    leading: Icon(Icons.card_membership),
                    title: Text('Membership'),
                    trailing: Icon(Icons.chevron_right)),
                const ListTile(
                    leading: Icon(Icons.straighten),
                    title: Text('Units'),
                    trailing: Text('kg',
                        style: TextStyle(color: AppColors.textDim))),
                ListTile(
                    onTap: _confirmSignOut,
                    leading: const Icon(Icons.logout, color: AppColors.bad),
                    title: const Text('Sign out',
                        style: TextStyle(color: AppColors.bad))),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _tile(
      String title, String sub, bool value, ValueChanged<bool> onChanged) {
    return SwitchListTile(
      title: Text(title),
      subtitle: Text(sub, style: const TextStyle(color: AppColors.textDim)),
      value: value,
      activeColor: AppColors.accent,
      onChanged: onChanged,
    );
  }
}
