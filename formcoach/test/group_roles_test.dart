import 'package:flutter_test/flutter_test.dart';
import 'package:formcoach/data/social_mock.dart';
import 'package:formcoach/models/social.dart';
import 'package:formcoach/screens/group_detail_screen.dart';
import 'package:formcoach/theme/app_theme.dart';
import 'package:flutter/material.dart';

/// Same members as "Leg Day Legends", seen from a chosen role.
Group legDayAs(GroupRole myRole) {
  final g = myGroups.first;
  return Group(
    id: g.id,
    name: g.name,
    description: g.description,
    isPrivate: g.isPrivate,
    isCommunity: g.isCommunity,
    myRole: myRole,
    joined: true,
    memberCount: g.memberCount,
    members: [
      const GroupMember(youssef, GroupRole.owner, 131),
      const GroupMember(sara, GroupRole.admin, 142),
      GroupMember(me, myRole == GroupRole.owner ? GroupRole.admin : myRole, 118),
      const GroupMember(omar, GroupRole.member, 96),
      const GroupMember(lina, GroupRole.member, 64),
    ],
  );
}

void main() {
  group('GroupRole rules', () {
    const o = GroupRole.owner, a = GroupRole.admin, m = GroupRole.member;

    test('only the owner changes roles, and never the owner role', () {
      expect(o.canChangeRoleOf(m), isTrue);
      expect(o.canChangeRoleOf(a), isTrue);
      expect(o.canChangeRoleOf(o), isFalse);
      for (final target in GroupRole.values) {
        expect(a.canChangeRoleOf(target), isFalse);
        expect(m.canChangeRoleOf(target), isFalse);
      }
    });

    test('owner removes anyone else, admin removes members only', () {
      expect(o.canRemove(a), isTrue);
      expect(o.canRemove(m), isTrue);
      expect(o.canRemove(o), isFalse);
      expect(a.canRemove(m), isTrue);
      expect(a.canRemove(a), isFalse);
      expect(a.canRemove(o), isFalse);
      for (final target in GroupRole.values) {
        expect(m.canRemove(target), isFalse);
      }
    });

    test('canManage is false only for members', () {
      expect(o.canManage, isTrue);
      expect(a.canManage, isTrue);
      expect(m.canManage, isFalse);
    });
  });

  group('GroupDetailScreen controls', () {
    Future<void> pumpGroup(WidgetTester tester, Group g) async {
      tester.view.physicalSize = const Size(1200, 6000);
      tester.view.devicePixelRatio = 3;
      addTearDown(tester.view.reset);
      await tester.pumpWidget(MaterialApp(
        theme: AppTheme.dark,
        home: GroupDetailScreen(group: g),
      ));
    }

    Future<void> openMenuFor(WidgetTester tester, String name) async {
      // The member list is below the leaderboard, so take the last match.
      await tester.tap(find.text(name).last);
      await tester.pumpAndSettle();
    }

    testWidgets('owner can promote, demote and remove', (tester) async {
      await pumpGroup(tester, myGroups.first); // you are the owner
      expect(find.text('Invite'), findsOneWidget);
      expect(find.text('Announce'), findsOneWidget);

      await openMenuFor(tester, 'Omar Nabil');
      expect(find.text('Make admin'), findsOneWidget);
      expect(find.text('Remove from group'), findsOneWidget);
      await tester.tap(find.text('Make admin'));
      await tester.pumpAndSettle();
      expect(find.text('ADMIN'), findsNWidgets(2)); // Sara and Omar

      await openMenuFor(tester, 'Sara Hassan');
      expect(find.text('Remove admin'), findsOneWidget);
      expect(find.text('Remove from group'), findsOneWidget);
      await tester.tap(find.text('Remove from group'));
      await tester.pumpAndSettle();
      expect(find.text('Members (4)'), findsOneWidget);
    });

    testWidgets('admin can only remove members', (tester) async {
      await pumpGroup(tester, legDayAs(GroupRole.admin));
      expect(find.text('Invite'), findsOneWidget);

      await openMenuFor(tester, 'Omar Nabil');
      expect(find.text('Make admin'), findsNothing);
      expect(find.text('Remove from group'), findsOneWidget);
      await tester.tapAt(const Offset(10, 10)); // dismiss the sheet
      await tester.pumpAndSettle();

      // Another admin and the owner have no actions for an admin.
      await openMenuFor(tester, 'Sara Hassan');
      expect(find.byType(BottomSheet), findsNothing);
      await openMenuFor(tester, 'Youssef Adel');
      expect(find.byType(BottomSheet), findsNothing);
      // Only the two plain members show the manage affordance.
      expect(find.byIcon(Icons.more_horiz), findsNWidgets(2));
    });

    testWidgets('member sees no management controls', (tester) async {
      await pumpGroup(tester, legDayAs(GroupRole.member));
      expect(find.text('Invite'), findsNothing);
      expect(find.text('Announce'), findsNothing);
      expect(find.text('Manage'), findsNothing);
      expect(find.text('Leave group'), findsOneWidget);
      expect(find.byIcon(Icons.more_horiz), findsNothing);

      await openMenuFor(tester, 'Omar Nabil');
      expect(find.byType(BottomSheet), findsNothing);
    });
  });
}
