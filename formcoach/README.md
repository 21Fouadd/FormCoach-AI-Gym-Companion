# FormCoach - Flutter UI template

UI-only template for a gym member app: live camera rep counting, form feedback,
video upload/record review, progress, profile. A mock analyzer drives every screen
so it runs end to end before your model is attached.

## Run
    flutter create --platforms=android,ios .   # generates android/ and ios/ folders (keeps lib/ and pubspec.yaml)
    flutter pub get
    flutter run

## Permissions (required for camera and video)
Android: android/app/src/main/AndroidManifest.xml
    <uses-permission android:name="android.permission.CAMERA"/>
    android/app/build.gradle: minSdkVersion 21 (camera plugin)
iOS: ios/Runner/Info.plist
    NSCameraUsageDescription, NSMicrophoneUsageDescription, NSPhotoLibraryUsageDescription

## Plugging in your model
Everything goes through lib/services/pose_analyzer.dart (abstract class PoseAnalyzer):
- analyzeLive(exercise): Stream<FrameAnalysis> -> reps, formScore (0-100), phase, issues, landmarks (0..1)
- analyzeVideo(path, exercise): Future<List<VideoFinding>> -> timestamped per-rep findings
Implement it, then change one line in lib/screens/shell.dart:
    final PoseAnalyzer analyzer = MyRealAnalyzer();
To feed live camera frames to your model, add controller.startImageStream(...) in
lib/screens/live_session_screen.dart (_initCamera) and push frames into your analyzer.
Update skeletonEdges in lib/models/models.dart to match your keypoint topology.

## Structure
lib/theme      design tokens + ThemeData
lib/models     Exercise, FrameAnalysis, VideoFinding, SessionResult
lib/services   PoseAnalyzer interface + mock
lib/widgets    Panel, ScoreRing, SeverityChip, SkeletonPainter, charts
lib/screens    shell (tabs), home, exercise picker, live session, summary, video analysis, progress, profile
lib/data       mock data (replace with your dataset/API)

## Social (added)
lib/models/social.dart        Person, Group, GroupMember, GroupRole (owner/admin/member), Post
lib/data/social_mock.dart     mock friends, groups, communities, feed
lib/screens/community_screen.dart      Feed / Friends / Groups tabs
lib/screens/group_detail_screen.dart   leaderboard, members, admin actions (invite, announce, promote, remove)
Role rules in the UI: owner can promote/demote admins and remove anyone; admins can remove members only; members see no management controls.
Enforce the same rules on the backend; the UI check is cosmetic.

## Accounts (added)
lib/services/auth_service.dart   AuthService interface, LocalAuthService, validators, AuthScope
lib/screens/auth/                login, sign up, forgot password, AuthGate (picks login or app)
- LocalAuthService stores accounts on the device (shared_preferences) with salted SHA-256 hashes.
  It is a prototype: no cross-device accounts and no real reset email.
- Demo account seeded on first run: demo@formcoach.app / formcoach123
- To move to a real backend, implement AuthService (e.g. SupabaseAuthService) and change one line
  in lib/main.dart: runApp(FormCoachApp(auth: SupabaseAuthService()));

## Building on Windows
- Build from a short path (for example S:\fc). Long paths break the shader compiler (260-char limit).
- android/gradle.properties sets kotlin.incremental=false because the pub cache is on another drive.
