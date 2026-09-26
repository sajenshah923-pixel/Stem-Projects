import unittest

from stem_projects.pushup import RepCounter, joint_angle


class PushupTests(unittest.TestCase):
    def test_angle_in_pixel_coordinates(self):
        self.assertAlmostEqual(joint_angle((0, 0), (1, 0), (1, 1)), 90)
        self.assertAlmostEqual(joint_angle((0, 0), (1, 0), (2, 0)), 180)
        with self.assertRaises(ValueError):
            joint_angle((0, 0), (0, 0), (1, 1))

    def test_full_clean_cycle_is_one_rep(self):
        counter = RepCounter()
        for angle in [175, 120, 85, 75, 120, 165, 175]:
            counter.update(angle, 175)
        self.assertEqual(counter.repetitions, 1)
        self.assertEqual(counter.rejected, 0)

    def test_bad_alignment_rejects_cycle(self):
        counter = RepCounter()
        counter.update(80, 170)
        self.assertEqual(counter.update(120, 130), "Keep body aligned")
        counter.update(170, 170)
        self.assertEqual((counter.repetitions, counter.rejected), (0, 1))


if __name__ == "__main__":
    unittest.main()

