# FormCoach - project brief for Claude Code

Graduation project: a gym member mobile app (Flutter) that counts reps and checks exercise form
from camera or video, plus social features (friends, groups with admins, public communities).

## Current state
- UI-only Flutter template in lib/. All data is mock (lib/data/*). Nothing has been compiled or run yet.
- docs/ui_reference.html is an interactive HTML mock of every screen. Open it in a browser: it is the visual source of truth.
- docs/design_tokens.json and lib/theme/app_theme.dart hold the design tokens (dark theme, lime accent). Keep them in sync.
- The model is NOT built. lib/services/pose_analyzer.dart (PoseAnalyzer) is the only integration point; MockPoseAnalyzer fakes it.

## First tasks, in order
1. Run `flutter create --platforms=android,ios .` in this folder (keep lib/ and pubspec.yaml), then `flutter pub get`.
2. Run `flutter analyze` and fix every error or deprecation. The code was written without a compiler.
   Version-sensitive spots: DropdownButtonFormField `value`, SwitchListTile `activeColor`, Color.withValues, camera 0.11 API.
3. Add camera/mic/photo permissions (Android manifest, iOS Info.plist), set Android minSdk 21.
4. Run on a device or emulator and walk every screen against docs/ui_reference.html. Fix layout overflows.
5. Add widget tests for the rep counter UI, the group role rules, and the score color rule.

## Model integration (after the user provides the video dataset)
- Pipeline: pose estimation (MediaPipe Pose or MoveNet) -> joint angles -> rep counting by peak/valley detection on one key angle per exercise -> form classifier on per-rep angle features.
- Training code lives in a separate folder ml/ (Python). Export the classifier to TFLite and load it in a RealPoseAnalyzer implementing PoseAnalyzer.
- Live: feed frames from CameraController.startImageStream into the analyzer; emit FrameAnalysis (reps, formScore, phase, issues, landmarks 0..1).
- Video: analyzeVideo(path, exercise) returns timestamped VideoFinding list.
- Update skeletonEdges in lib/models/models.dart to match the keypoint topology used.
- Ask the user: which exercises, label format (correct/incorrect or specific mistakes), clip count, camera angle.

## Backend (not started)
- Suggested: Supabase (auth, Postgres, storage for clips, row level security).
- Tables: profiles, friendships(status), groups(is_private, is_community), group_members(role owner|admin|member), posts, post_likes, sessions, session_findings.
- Role rules the UI already shows: owner can promote/demote admins and remove anyone; admin can invite, announce, remove members only; member has no management controls. Enforce these with RLS policies, not only in the UI.
- Replace lib/data/*_mock.dart with repositories behind interfaces so the UI does not change.

## Conventions
- No emoji anywhere in UI or code.
- Use design tokens (AppColors, AppSpace, AppRadius), never raw hex in screens.
- Form status colors: score >= 85 ok, 70-84 warn, < 70 bad; always pair color with an icon or label.
- One primary (lime) button per screen.
- Keep the PoseAnalyzer interface stable; add new fields as optional.
