import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/api_service.dart';

class RegisterUserScreen extends StatefulWidget {
  const RegisterUserScreen({super.key});

  @override
  State<RegisterUserScreen> createState() => _RegisterUserScreenState();
}

class _RegisterUserScreenState extends State<RegisterUserScreen> {
  final _nameCtrl = TextEditingController();
  final _ageCtrl = TextEditingController(text: '22');
  final _heightCtrl = TextEditingController(text: '178');
  final _weightCtrl = TextEditingController(text: '75');
  String _fitnessLevel = 'Auto';
  File? _photo;
  bool _submitting = false;
  List<Map<String, dynamic>> _users = [];

  @override
  void initState() {
    super.initState();
    _refreshUsers();
  }

  Future<void> _refreshUsers() async {
    try {
      final list = await ApiService.listUsers();
      if (mounted) setState(() => _users = list);
    } catch (_) {}
  }

  Future<void> _pickPhoto(ImageSource source) async {
    final picked = await ImagePicker().pickImage(source: source, maxWidth: 800, imageQuality: 85);
    if (picked != null) setState(() => _photo = File(picked.path));
  }

  Future<void> _submit() async {
    if (_nameCtrl.text.trim().isEmpty || _photo == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please enter a name and take/pick a front face photo.')),
      );
      return;
    }
    setState(() => _submitting = true);
    try {
      await ApiService.registerUser(
        name: _nameCtrl.text.trim(),
        age: int.tryParse(_ageCtrl.text) ?? 22,
        heightCm: double.tryParse(_heightCtrl.text) ?? 175,
        weightKg: double.tryParse(_weightCtrl.text) ?? 70,
        fitnessLevel: _fitnessLevel,
        photoFile: _photo!,
      );
      _nameCtrl.clear();
      setState(() => _photo = null);
      await _refreshUsers();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Athlete registered with Face ID!')),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: $e')));
      }
    } finally {
      if (mounted) setState(() => _submitting = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        const Text('Register Athlete (Face ID)', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
        const SizedBox(height: 12),
        TextField(controller: _nameCtrl, decoration: const InputDecoration(labelText: 'Full Name')),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(child: TextField(controller: _ageCtrl, keyboardType: TextInputType.number, decoration: const InputDecoration(labelText: 'Age'))),
            const SizedBox(width: 10),
            Expanded(child: TextField(controller: _heightCtrl, keyboardType: TextInputType.number, decoration: const InputDecoration(labelText: 'Height (cm)'))),
            const SizedBox(width: 10),
            Expanded(child: TextField(controller: _weightCtrl, keyboardType: TextInputType.number, decoration: const InputDecoration(labelText: 'Weight (kg)'))),
          ],
        ),
        const SizedBox(height: 10),
        DropdownButtonFormField<String>(
          value: _fitnessLevel,
          decoration: const InputDecoration(labelText: 'Fitness Level'),
          items: const [
            DropdownMenuItem(value: 'Auto', child: Text('Auto (AI Predicted via SmartFit)')),
            DropdownMenuItem(value: 'Beginner', child: Text('Beginner')),
            DropdownMenuItem(value: 'Intermediate', child: Text('Intermediate')),
            DropdownMenuItem(value: 'Advanced', child: Text('Advanced')),
          ],
          onChanged: (v) => setState(() => _fitnessLevel = v ?? 'Auto'),
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            ElevatedButton.icon(
              onPressed: () => _pickPhoto(ImageSource.camera),
              icon: const Icon(Icons.camera_alt),
              label: const Text('Take Face Photo'),
            ),
            const SizedBox(width: 10),
            OutlinedButton.icon(
              onPressed: () => _pickPhoto(ImageSource.gallery),
              icon: const Icon(Icons.photo),
              label: const Text('Gallery'),
            ),
          ],
        ),
        if (_photo != null) ...[
          const SizedBox(height: 10),
          ClipRRect(
            borderRadius: BorderRadius.circular(10),
            child: Image.file(_photo!, height: 150, fit: BoxFit.cover),
          ),
        ],
        const SizedBox(height: 14),
        FilledButton(
          onPressed: _submitting ? null : _submit,
          child: Text(_submitting ? 'Extracting Face Embedding...' : 'Register Athlete'),
        ),
        const Divider(height: 32),
        Text('Registered Athletes (${_users.length})', style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
        const SizedBox(height: 8),
        for (final u in _users)
          Card(
            child: ListTile(
              leading: const CircleAvatar(child: Icon(Icons.person)),
              title: Text('${u['name']} (${u['fitness_level']})'),
              subtitle: Text('Age ${u['age']} • ${u['height_cm']} cm • ${u['weight_kg']} kg'),
              trailing: IconButton(
                icon: const Icon(Icons.delete_outline, color: Colors.redAccent),
                onPressed: () async {
                  await ApiService.deleteUser(u['id'] as int);
                  _refreshUsers();
                },
              ),
            ),
          ),
      ],
    );
  }
}
