import '../models/social.dart';

const sara = Person(id: 'u2', name: 'Sara Hassan', streak: 12, avgForm: 91);
const omar = Person(id: 'u3', name: 'Omar Nabil', streak: 7, avgForm: 84);
const lina = Person(id: 'u4', name: 'Lina Fouad', streak: 3, avgForm: 77);
const youssef = Person(id: 'u5', name: 'Youssef Adel', streak: 21, avgForm: 88);
const me = Person(id: 'u1', name: 'Alex Morgan', streak: 5, avgForm: 86);

const requests = <Person>[
  Person(
      id: 'u6',
      name: 'Hana Samir',
      streak: 4,
      avgForm: 80,
      status: FriendStatus.requestReceived),
];

const suggestions = <Person>[
  Person(
      id: 'u7',
      name: 'Karim Wael',
      streak: 9,
      avgForm: 82,
      status: FriendStatus.none),
  Person(
      id: 'u8',
      name: 'Nour Tarek',
      streak: 2,
      avgForm: 75,
      status: FriendStatus.none),
];

const friends = <Person>[sara, omar, lina, youssef];

const myGroups = <Group>[
  Group(
    id: 'g1',
    name: 'Leg Day Legends',
    description: 'Squats, lunges and deadlifts. Share your sets, fix your form together.',
    isPrivate: true,
    isCommunity: false,
    myRole: GroupRole.owner,
    joined: true,
    memberCount: 6,
    members: [
      GroupMember(me, GroupRole.owner, 118),
      GroupMember(sara, GroupRole.admin, 142),
      GroupMember(omar, GroupRole.member, 96),
      GroupMember(youssef, GroupRole.member, 131),
      GroupMember(lina, GroupRole.member, 64),
    ],
  ),
  Group(
    id: 'g2',
    name: 'Campus Gym Crew',
    description: 'University gym buddies. Weekly challenges every Sunday.',
    isPrivate: true,
    isCommunity: false,
    myRole: GroupRole.member,
    joined: true,
    memberCount: 14,
    members: [
      GroupMember(omar, GroupRole.owner, 110),
      GroupMember(sara, GroupRole.admin, 142),
      GroupMember(me, GroupRole.member, 118),
    ],
  ),
];

const communities = <Group>[
  Group(
    id: 'c1',
    name: 'Beginner Form Clinic',
    description: 'Post a clip, get feedback. New lifters welcome.',
    isPrivate: false,
    isCommunity: true,
    myRole: GroupRole.member,
    joined: false,
    memberCount: 1280,
    members: [],
  ),
  Group(
    id: 'c2',
    name: 'Calisthenics Cairo',
    description: 'Push-ups, pull-ups, dips. Meetups and weekly rep challenges.',
    isPrivate: false,
    isCommunity: true,
    myRole: GroupRole.member,
    joined: false,
    memberCount: 642,
    members: [],
  ),
];

const feed = <Post>[
  Post(
    author: sara,
    when: '2h ago',
    text: 'New squat PB with clean form. Knees finally stopped caving.',
    exercise: 'Back Squat',
    reps: 12,
    formScore: 94,
    likes: 18,
    groupName: 'Leg Day Legends',
  ),
  Post(
    author: youssef,
    when: '5h ago',
    text: '21 day streak. Showing up beats motivation.',
    exercise: 'Push-up',
    reps: 50,
    formScore: 88,
    likes: 32,
  ),
  Post(
    author: omar,
    when: 'Yesterday',
    text: 'Deadlift form check please. Is my back rounding at the bottom?',
    exercise: 'Deadlift',
    reps: 5,
    formScore: 71,
    likes: 6,
    groupName: 'Beginner Form Clinic',
  ),
];
