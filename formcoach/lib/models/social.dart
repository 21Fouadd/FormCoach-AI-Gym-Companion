enum GroupRole { owner, admin, member }

extension GroupRoleX on GroupRole {
  String get label => switch (this) {
        GroupRole.owner => 'Owner',
        GroupRole.admin => 'Admin',
        GroupRole.member => 'Member',
      };
  bool get canManage => this != GroupRole.member;

  /// Only the owner can promote a member to admin or demote an admin.
  bool canChangeRoleOf(GroupRole target) =>
      this == GroupRole.owner && target != GroupRole.owner;

  /// Owner can remove anyone but themselves; admins can remove members only.
  bool canRemove(GroupRole target) => switch (this) {
        GroupRole.owner => target != GroupRole.owner,
        GroupRole.admin => target == GroupRole.member,
        GroupRole.member => false,
      };

  /// True when [this] has at least one action on a member with [target] role.
  bool hasActionsOn(GroupRole target) =>
      canChangeRoleOf(target) || canRemove(target);
}

enum FriendStatus { friend, requestReceived, requestSent, none }

class Person {
  final String id;
  final String name;
  final int streak;
  final double avgForm;
  final FriendStatus status;

  const Person({
    required this.id,
    required this.name,
    required this.streak,
    required this.avgForm,
    this.status = FriendStatus.friend,
  });

  String get initials => name.split(' ').map((w) => w[0]).take(2).join();
}

class GroupMember {
  final Person person;
  final GroupRole role;
  final int weeklyReps;
  const GroupMember(this.person, this.role, this.weeklyReps);
}

class Group {
  final String id;
  final String name;
  final String description;
  final bool isPrivate;
  final bool isCommunity; // public, discoverable
  final GroupRole myRole; // your role, or member if not joined
  final bool joined;
  final List<GroupMember> members;
  final int memberCount;

  const Group({
    required this.id,
    required this.name,
    required this.description,
    required this.isPrivate,
    required this.isCommunity,
    required this.myRole,
    required this.joined,
    required this.members,
    required this.memberCount,
  });
}

class Post {
  final Person author;
  final String when;
  final String text;
  final String? exercise;
  final int? reps;
  final double? formScore;
  final int likes;
  final String? groupName;

  const Post({
    required this.author,
    required this.when,
    required this.text,
    this.exercise,
    this.reps,
    this.formScore,
    required this.likes,
    this.groupName,
  });
}
