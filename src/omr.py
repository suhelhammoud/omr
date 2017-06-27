import omr_border as brd
from omr_border import *
from omr_border import _get_markers, _marker_filter, _update_markers_with_x, _draw_markers_lines
from omr_markers import process_marker_column, adjust_markers_y_shift, draw_h_lines_on_markers
from omr_answer import get_answers, Answer
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    print('starting processing')

    file_path = '../data/in2/05.jpg'
    img = cv2.imread(file_path, 0)
    # plt.imshow(img, 'gray')
    # plt.show()
    height, width = img.shape
    # logger.debug("height %s, width %s", height, width)

    sheet = get_sheet(img, debug=False)

    vis_sheet = sheet.copy()

    sheet_type_img = conf.sec_type.crop(sheet)
    sheet_type = process_type(sheet_type_img, debug=False)
    sheet_id_img = conf.sec_id.crop(sheet)
    sheet_id_number = process_id(sheet_id_img, debug=False)
    # assert sheet_id_number == 7036093052

    sheet_otsu = otsu_filter(sheet, blur_kernel=None)
    markers = process_marker_column(sheet_otsu, conf.sec_marker_column, debug=False)

    draw_h_lines_on_markers(markers, vis_sheet)

    vis_sheet2 = sheet.copy()

    m_50 = {(m.id - 12): (m.x2 + 2, m.center_y_int())
            for m in markers if m.id > 12}

    # m_100 =
    answers = get_answers(sheet_otsu, m_50)

    answers[1] = Answer.next(answers[5], answers[2])
    answers[50] = Answer.next(answers[40], answers[49])

    for num, answer in answers.items():
        logger.info('answer no: %s, data: %s', num, answer)
        answer.draw_lines(vis_sheet2)

    # adjust_markers_y_shift(markers, sheet_otsu, conf.marker_calibre_range, debug=False)


    # draw_h_lines_on_markers(markers, vis_sheet2)

    plt.subplot(121), plt.imshow(vis_sheet, 'gray'), plt.title('no y shift')
    plt.subplot(122), plt.imshow(vis_sheet2, 'gray'), plt.title('with y shift')
    plt.show()

    exit(0)

    # sheet_marker = _marker_filter(sheet_marker,
    #                               blur_param=(14, 1),
    #                               median_param=conf.marker_filter_median_blur)

    exit(0)

    y_sum = sheet_markers_img.sum(1)
    # y_sum = sheet_marker.sum(1) / conf.marker_r_shift
    logger.debug('max_sum = %s', y_sum[np.argmax(y_sum)])
    # y_avg = np.average(y_sum)

    avg_smoothed = smooth(y_sum, window_len=conf.marker_smooth_window, window='flat')

    logger.debug("y_sum shape = %s ", y_sum.shape)
    logger.debug("avg_s shape = %s ", avg_smoothed.shape)

    markers_list = _get_markers(y_sum, avg_smoothed, conf.marker_threshold_spacing)
    if len(markers_list) != 63:
        plt.subplot(313), plt.plot(y_sum, 'r', avg_smoothed, 'b'), plt.title(
            "error ")  # very important to debug splitting points
        plt.show()
    # logger.debug('markers: %s', markers)
    # avg_m = avg_marker_height(markers)
    # logger.debug('avg marker height = %s, %s', avg_m, math.ceil(avg_m))

    # section_marker_top = conf.top_marker.translate(0, markers[0].y0)

    markers = _update_markers_with_x(markers_list, sheet)

    last_shift = 0

    for m in markers.values():
        if m.id in [15, 29, 33, 37, 41, 47, 49]:
            last_shift = brd._calibrate_with_marker(m, sheet, debug=True)
        m.set_shift_y(last_shift)

    # for i in [15, 29, 33, 37, 41, 47, 49]:
    #     diff = calibrate_with_marker(markers[i], sheet)
    #     markers[i].set_shift_y(diff)
    #     logger.debug('marker %s shift = %s', i, markers[i].shift_y)

    _draw_markers_lines(vis_sheet, markers)
    # section_markers = [conf.sec_marker.translate(0, m.y0) for m in markers[:]]
    # # marker_top = section_marker_top.crop(sheet)
    # for sec_marker, marker in zip(section_markers, markers):
    #     marker_roi = sec_marker.crop(sheet)
    #     x0, x1 = get_marker_x0_x1(marker_roi)
    #     marker.set_x0_x1(x0, x1)

    # plt.subplot(311), plt.imshow(marker_roi, 'gray'), plt.title("marker_roi")
    # plt.subplot(312), plt.plot(sum_marker, 'r'), plt.title("marker_sum")
    # plt.subplot(313), plt.plot(y_sum, 'r', avg_smoothed, 'b')  # very important to debug splitting points
    # plt.show()

    # for i in avg_smoothed:
    #     print(i)

    # for i in y_sum:
    #     print(i)

    # for yy, x_img in slide_marker(sheet_marker, conf.y_step, conf.y_window):
    #     plt.imshow(x_img, 'gray'), plt.title(str(yy))
    #     plt.show()
    #
    # for m in markers:
    #     y0, y1 = m.y0_y1()
    #     x0, x1 = m.x0_x1()
    #     cv2.line(sheet, (0, y0), (width, y0), (0, 255, 255), 1)
    #     cv2.line(sheet, (0, y1), (width, y1), (255, 255, 255), 1)
    #     cv2.line(sheet, (x0, y0), (x0, y1), (255, 255, 255), 1)
    #     cv2.line(sheet, (x1, y0), (x1, y1), (255, 255, 255), 1)

    plt.subplot(231), plt.imshow(img, 'gray'), plt.title('Input')
    plt.subplot(232), plt.imshow(vis_sheet, 'gray'), plt.title('Sheet')
    plt.subplot(233), plt.imshow(sheet_markers_img, 'gray'), plt.title('sheet_marker')
    plt.subplot(234), plt.imshow(sheet_id_img, 'gray'), plt.title('Name')
    plt.subplot(235), plt.imshow(sheet_type_img, 'gray'), plt.title('type')
    plt.subplot(236), plt.imshow(sheet_one, 'gray'), plt.title('one')
    plt.show()
